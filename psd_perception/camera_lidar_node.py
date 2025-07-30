#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2, LaserScan
from psd_msgs.msg import ConePosition, ConePositionArrayStamped, ConeRangeBearing, ConeRangeBearingArrayStamped
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from rcl_interfaces.msg import ParameterDescriptor

from cv_bridge import CvBridge
import numpy as np
import cv2
import pycuda.autoinit  # Initializes CUDA driver
import tensorrt as trt
import pycuda.driver as cuda
import struct

import threading
import csv
import os
from datetime import datetime
from collections import deque
import time


class ConeDetectorNode(Node):
	def __init__(self):
		super().__init__('cone_detector_node')
		
		# Declare parameters
		self.declare_parameter('engine_path', '/home/psd/psd_ws/src/psd_perception/best.engine',
							  ParameterDescriptor(description='Path to TensorRT engine file'))
		
		self.declare_parameter('conf_threshold', 0.5,
							  ParameterDescriptor(description='Confidence threshold for detection'))
		
		self.declare_parameter('input_size', 640,
							  ParameterDescriptor(description='Model input size'))
		
		self.declare_parameter('debug', False,
							  ParameterDescriptor(description='Enable debug mode to save detections to CSV'))
		
		self.declare_parameter('debug_csv_path', 'cone_detections_debug.csv',
							  ParameterDescriptor(description='Path to save debug CSV file'))
		
		self.declare_parameter('centroid_mode', 1,
							  ParameterDescriptor(description='Centroid computation mode: 0=all points, 1=color filter, 2=center only'))
		
		self.declare_parameter('filter', False,
							  ParameterDescriptor(description='Enable EMA filtering for centroids'))
		
		self.declare_parameter('ema_alpha', 0.3,
							  ParameterDescriptor(description='EMA filter alpha value (0-1, higher=more responsive)'))
		
		# Get parameters
		self.engine_path = self.get_parameter('engine_path').value
		self.conf_threshold = self.get_parameter('conf_threshold').value
		self.input_size = self.get_parameter('input_size').value
		self.debug_mode = self.get_parameter('debug').value
		self.debug_csv_path = self.get_parameter('debug_csv_path').value
		self.centroid_mode = self.get_parameter('centroid_mode').value
		self.filter_enabled = self.get_parameter('filter').value
		self.alpha = self.get_parameter('ema_alpha').value

		# Range constraints for detection in pcl
		self.minRange = 0.1
		self.maxRange = 20.0 
		
		# Initialize debug CSV if debug mode is enabled
		if self.debug_mode:
			self.init_debug_csv()
			self.get_logger().info(f"Debug mode enabled. Saving detections to: {self.debug_csv_path}")
			# Pre-open CSV file for performance
			self.debug_csv_file = open(self.debug_csv_path, 'a', newline='')
			self.debug_csv_writer = csv.writer(self.debug_csv_file)
		
		# Initialize TensorRT - store in a dict to avoid attribute conflicts
		self.trt_components = {}
		try:
			self.get_logger().info(f"Loading engine {self.engine_path} ...")
			engine, context = self.load_engine(self.engine_path)
			self.trt_components['engine'] = engine
			self.trt_components['context'] = context
			
			self.get_logger().info("Allocating buffers ...")
			(self.host_inputs, self.cuda_inputs, self.input_tensor_names,
			 self.host_outputs, self.cuda_outputs, self.output_tensor_names, 
			 self.bindings) = self.allocate_buffers(self.trt_components['engine'])
			
			self.output_tensor_name = self.output_tensor_names[0]

			self.preprocess_buffer = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
			
		except Exception as e:
			self.get_logger().error(f"Failed to initialize TensorRT: {str(e)}")
			raise
		
		# Initialize CV Bridge
		self.bridge = CvBridge()
		
		# Storage for latest data
		self.latest_image = None
		self.latest_pointcloud = None
		self.latest_image_msg = None
		self.latest_image_stamp = None
		self.latest_pointcloud_stamp = None
		
		# EMA centroid storage - improved tracking
		self._ema_centroids = {}  # Key: (class_id, approx_x, approx_y) -> Value: (centroid, last_update_time)
		self.ema_timeout = 0.5  # Remove EMA entries older than 500ms
		self.ema_spatial_threshold = 1.0  # Max distance in meters to match previous centroid
		
		# Processing control
		self.processing_lock = threading.Lock()
		self.processing_in_progress = False
		self.last_process_time = 0
		self.min_process_interval = 0.033  # ~30Hz max processing rate
		
		# Define cone class colors (adjust based on your FSOCO classes)
		self.class_colors = {
			7: (255, 0, 0),      # Blue cone
			2: (0, 165, 255),    # Yellow cone
		}
		self.class_names = {
			7: "Blue",
			2: "Yellow", 
		}
		
		# Color tolerance for centroid computation
		self.color_tolerance = 60
		
		# Performance tracking
		self.performance_window = deque(maxlen=30)
		
		# Subscribers
		self.image_sub = self.create_subscription(
			Image,
			'/zed/zed_node/left/image_rect_color',
			self.image_callback,
			10
		)
		
		self.pointcloud_sub = self.create_subscription(
			PointCloud2,
			'/zed/zed_node/point_cloud/cloud_registered',
			self.pointcloud_callback,
			10
		)

		self.lidar_sub = self.create_subscription(
			LaserScan,
			'/scan',
			self.lidar_callback,
			10
		)
		
		# Publishers
		self.cone_publisher = self.create_publisher(
			PointCloud2,
			'/possible_cones',
			10
		)

		self.cone_position_publisher = self.create_publisher(
			ConePositionArrayStamped,
			'/possible_cones_xyz',
			10
		)
		
		self.bbox_image_publisher = self.create_publisher(
			Image,
			'/detected_bb',
			10
		)

		self.viewed_cone_publisher = self.create_publisher(
			ConePositionArrayStamped,
			'/viewed_cones_xyz',
			10
		)
		
		self.get_logger().info('Cone Detector Node initialized successfully')
		self.get_logger().info(f'Publishing bounding box visualization to /detected_bb')
		self.get_logger().info(f'Publishing cone positions to /possible_cones and /possible_cones_xyz')
		self.get_logger().info(f'EMA filter: {"Enabled" if self.filter_enabled else "Disabled"}, Alpha: {self.alpha}')
	
	def __del__(self):
		"""Cleanup resources"""
		if hasattr(self, 'debug_csv_file'):
			self.debug_csv_file.close()
	
	def init_debug_csv(self):
		"""Initialize CSV file for debug logging"""
		try:
			# Create directory if it doesn't exist
			csv_dir = os.path.dirname(self.debug_csv_path)
			if csv_dir and not os.path.exists(csv_dir):
				os.makedirs(csv_dir)
			
			# Write headers if file doesn't exist
			if not os.path.exists(self.debug_csv_path):
				with open(self.debug_csv_path, 'w', newline='') as f:
					writer = csv.writer(f)
					writer.writerow([
						'timestamp', 'ros_time', 'detection_id', 'class_id', 'class_name',
						'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
						'centroid_x', 'centroid_y', 'centroid_z', 'distance_m',
						'inference_time_ms', 'centroid_time_ms', 'total_time_ms'
					])
		except Exception as e:
			self.get_logger().error(f"Failed to initialize debug CSV: {str(e)}")
			self.debug_mode = False
	
	def save_debug_detections(self, detections, centroids, distances, inference_time_ms, centroid_time_ms):
		"""Save detection results to CSV file - optimized version"""
		if not self.debug_mode:
			return
		
		try:
			timestamp = datetime.now().isoformat()
			ros_time = self.get_clock().now().nanoseconds
			total_time_ms = inference_time_ms + centroid_time_ms
			
			# Write each detection
			for i, detection in enumerate(detections):
				bbox = detection['bbox']
				class_id = detection['class_id']
				confidence = detection['confidence']
				class_name = self.class_names.get(class_id, f"Class_{class_id}")
				
				# Get corresponding centroid if available
				if i < len(centroids) and centroids[i] is not None:
					cx, cy, cz = centroids[i]
					cx_str, cy_str, cz_str = f"{cx:.4f}", f"{cy:.4f}", f"{cz:.4f}"
				else:
					cx_str, cy_str, cz_str = 'N/A', 'N/A', 'N/A'
				
				# Get distance
				dist_str = f"{distances[i]:.2f}" if i < len(distances) and distances[i] is not None else 'N/A'
				
				self.debug_csv_writer.writerow([
					timestamp, ros_time, i, class_id, class_name,
					f"{confidence:.4f}", bbox[0], bbox[1], bbox[2], bbox[3],
					cx_str, cy_str, cz_str, dist_str,
					f"{inference_time_ms:.2f}",
					f"{centroid_time_ms:.2f}",
					f"{total_time_ms:.2f}"
				])
			
			# Flush periodically to ensure data is written
			if len(detections) > 0:
				self.debug_csv_file.flush()
			
		except Exception as e:
			self.get_logger().error(f"Failed to save debug data: {str(e)}")
	
	def load_engine(self, engine_path):
		"""Load TensorRT engine from file"""
		logger = trt.Logger(trt.Logger.WARNING)
		with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
			engine = runtime.deserialize_cuda_engine(f.read())
		if not engine:
			raise RuntimeError("Failed to deserialize TensorRT engine.")
		execution_context = engine.create_execution_context()
		if not execution_context:
			raise RuntimeError("Failed to create TensorRT execution context.")
		return engine, execution_context
	
	def allocate_buffers(self, engine, batch_size=1):
		"""Allocates host and device buffers for all I/O tensors of the engine."""
		num_tensors = engine.num_io_tensors
		bindings = [None] * num_tensors
		host_inputs = []
		cuda_inputs = []
		input_tensor_names = []
		host_outputs = []
		cuda_outputs = []
		output_tensor_names = []
		
		for i in range(num_tensors):
			tensor_name = engine.get_tensor_name(i)
			shape = list(engine.get_tensor_shape(tensor_name))
			if shape[0] == -1:
				shape[0] = batch_size
			size = trt.volume(shape)
			dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
			host_mem = np.zeros(size, dtype=dtype)
			cuda_mem = cuda.mem_alloc(host_mem.nbytes)
			bindings[i] = int(cuda_mem)
			
			if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
				host_inputs.append(host_mem)
				cuda_inputs.append(cuda_mem)
				input_tensor_names.append(tensor_name)
			else:
				host_outputs.append(host_mem)
				cuda_outputs.append(cuda_mem)
				output_tensor_names.append(tensor_name)
		
		return host_inputs, cuda_inputs, input_tensor_names, host_outputs, cuda_outputs, output_tensor_names, bindings
	
	def preprocess_image(self, image_bgr):
		"""Optimized preprocessing"""
		# Use cv2.resize with INTER_LINEAR for speed
		image_resized = cv2.resize(image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
		
		# Convert BGR to RGB and normalize in one step
		image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0/255.0)
		
		# Transpose and expand dims
		image_transposed = np.transpose(image_rgb, (2, 0, 1))
		self.preprocess_buffer[0] = image_transposed
		
		return self.preprocess_buffer
	
	def do_inference(self, input_data):
		"""Run inference - optimized"""
		# Copy preprocessed input data to host input buffer
		np.copyto(self.host_inputs[0], input_data.ravel())
		# Transfer input from host to device
		cuda.memcpy_htod(self.cuda_inputs[0], self.host_inputs[0])
		# Execute inference
		self.trt_components['context'].execute_v2(bindings=self.bindings)
		# Transfer output from device back to host
		cuda.memcpy_dtoh(self.host_outputs[0], self.cuda_outputs[0])
		# Retrieve output tensor shape
		out_shape = list(self.trt_components['engine'].get_tensor_shape(self.output_tensor_name))
		out_shape = [1 if d == -1 else d for d in out_shape]
		return self.host_outputs[0].reshape(out_shape)
	
	def postprocess_detections(self, predictions, orig_image_shape):
		"""Optimized postprocessing"""
		preds = predictions[0]
		h_orig, w_orig = orig_image_shape[:2]
		
		# Vectorized operations
		scale_x = w_orig / self.input_size
		scale_y = h_orig / self.input_size
		
		# Filter by confidence threshold
		valid_mask = preds[:, 4] >= self.conf_threshold
		valid_preds = preds[valid_mask]
		
		results = []
		for pred in valid_preds:
			x1 = int(pred[0] * scale_x)
			y1 = int(pred[1] * scale_y)
			x2 = int(pred[2] * scale_x)
			y2 = int(pred[3] * scale_y)
			
			# Clip coordinates
			x1 = np.clip(x1, 0, w_orig - 1)
			y1 = np.clip(y1, 0, h_orig - 1)
			x2 = np.clip(x2, 0, w_orig - 1)
			y2 = np.clip(y2, 0, h_orig - 1)
			
			results.append({
				'bbox': [x1, y1, x2, y2],
				'confidence': float(pred[4]),
				'class_id': int(pred[5])
			})
		
		return results
	
	def apply_ema_filter(self, raw_centroid, class_id):
		"""Apply improved EMA filter with spatial tracking"""
		if not self.filter_enabled or raw_centroid is None:
			return raw_centroid
		
		current_time = time.time()
		
		# Clean up old EMA entries
		self._ema_centroids = {
			k: v for k, v in self._ema_centroids.items() 
			if current_time - v[1] < self.ema_timeout
		}
		
		# Find closest previous centroid of same class
		best_match = None
		best_distance = float('inf')
		
		for key, (prev_centroid, _) in self._ema_centroids.items():
			if key[0] == class_id:  # Same class
				distance = np.linalg.norm(raw_centroid - prev_centroid)
				if distance < best_distance and distance < self.ema_spatial_threshold:
					best_distance = distance
					best_match = key
		
		# Apply EMA if match found
		if best_match is not None:
			prev_centroid, _ = self._ema_centroids[best_match]
			filtered_centroid = self.alpha * raw_centroid + (1 - self.alpha) * prev_centroid
			# Update with new position
			self._ema_centroids[best_match] = (filtered_centroid, current_time)
		else:
			# Create new tracking entry
			filtered_centroid = raw_centroid
			# Use approximate grid position as key to handle small movements
			approx_x = round(raw_centroid[0], 1)
			approx_y = round(raw_centroid[1], 1)
			key = (class_id, approx_x, approx_y)
			self._ema_centroids[key] = (filtered_centroid, current_time)
		
		return filtered_centroid
	
	def draw_boxes(self, image, detections, centroid_points=None, centroids=None):
		"""Draw bounding boxes, centroid points, and distance information on the image"""
		annotated = image.copy()
		
		# Draw centroid points first (so boxes are on top)
		if centroid_points is not None:
			for points in centroid_points:
				if points is not None:
					for u, v in points:
						# Draw red circles for centroid computation points
						cv2.circle(annotated, (int(u), int(v)), 1, (0, 0, 255), -1)
						# Add white border for visibility
						cv2.circle(annotated, (int(u), int(v)), 2, (255, 255, 255), 1)
		
		# Draw bounding boxes with distance info
		for i, detection in enumerate(detections):
			x1, y1, x2, y2 = detection['bbox']
			conf = detection['confidence']
			cls_id = detection['class_id']
			
			# Get color and name for this class
			color = self.class_colors.get(cls_id, (0, 255, 0))
			class_name = self.class_names.get(cls_id, f"Class_{cls_id}")
			
			# Draw bounding box
			cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
			
			# Calculate distance if centroid available
			distance_text = ""
			if centroids is not None and i < len(centroids) and centroids[i] is not None:
				distance = np.linalg.norm(centroids[i])  # Euclidean distance from camera
				distance_text = f" | {distance:.2f}m"
			
			# Draw label with background for better visibility
			label = f"{class_name}: {conf:.2f}{distance_text}"
			label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
			cv2.rectangle(annotated, (x1, max(0, y1-label_size[1]-10)), 
						 (x1 + label_size[0], y1), color, -1)
			cv2.putText(annotated, label, (x1, max(0, y1-5)),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
		
		# Add detection count and mode info
		info_text = f"Detected {len(detections)} cone(s) \nMode: {self.centroid_mode}"
		if self.filter_enabled:
			info_text += f" | EMA: {self.alpha}"
		cv2.putText(annotated, info_text, 
				   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
				   1.0, (255, 255, 255), 1)
		
		return annotated
	
	def compute_bbox_centroid(self, pointcloud, bbox, detection):
		"""
		Optimized centroid extraction with visualization support.
		Returns (centroid, points_used) where points_used is list of (u,v) coordinates
		"""
		x1, y1, x2, y2 = bbox
		class_id = detection['class_id']
		
		try:
			# For organized point clouds from ZED
			if pointcloud.height > 1 and pointcloud.width > 1:
				# Build list of UV coordinates within bounding box
				# Adaptive step size based on bbox size
				# Compute isosceles triangle inside bbox
				bbox_w = x2 - x1
				bbox_h = y2 - y1
				if bbox_w <= 0 or bbox_h <= 0:
					return None, None

				# Scale factor: 1.0 means base equals full bbox width (isosceles triangle)
				scale = 1.0  # You can make this a parameter if needed

				# Top vertex (middle of top side)
				top_u = int(x1 + bbox_w / 2)
				top_v = y1

				# Bottom vertices (left and right of bottom side, full width)
				left_u = int(x1)
				right_u = int(x2)
				bottom_v = y2

				# Clamp to image/pointcloud bounds
				left_u = max(0, min(left_u, pointcloud.width - 1))
				right_u = max(0, min(right_u, pointcloud.width - 1))
				top_u = max(0, min(top_u, pointcloud.width - 1))

				# Rasterize triangle area (simple scanline fill)
				uvs = []
				for v in range(y1, y2 + 1):
					if bbox_h == 0:
						continue
					rel_v = (v - y1) / bbox_h  # 0 at top, 1 at bottom

					# Interpolate left/right bounds for this row (isosceles triangle)
					curr_left_u = int(top_u + (left_u - top_u) * rel_v)
					curr_right_u = int(top_u + (right_u - top_u) * rel_v)

					# Clamp
					curr_left_u = max(0, min(curr_left_u, pointcloud.width - 1))
					curr_right_u = max(0, min(curr_right_u, pointcloud.width - 1))

					# Step size: coarser for large bboxes
					step = 10 if bbox_w * bbox_h > 10000 else 5 if bbox_w * bbox_h > 5000 else 1
					for u in range(curr_left_u, curr_right_u + 1, step):
						if 0 <= u < pointcloud.width and 0 <= v < pointcloud.height:
							uvs.extend([u, v])

				if not uvs:
					return None, None
				
				points = []
				points_used = []  # Store (u,v) coordinates for visualization

				# Mode 3: Horizontal centerline, 90th percentile of points
				if self.centroid_mode == 3:
					center_v = int((y1 + y2) / 2)
					u_start = max(0, x1)
					u_end = min(pointcloud.width - 1, x2)
					points = []
					points_used = []
					for u in range(u_start, u_end + 1):
						p_dbg = pc2.read_points(pointcloud, field_names=['x', 'y', 'z'], skip_nans=False)
						idx = int(center_v * pointcloud.width + u)
						point_gen = [p_dbg[idx]]
						point_list = list(point_gen)
						if point_list and len(point_list[0]) >= 3:
							x, y, z = float(point_list[0][0]), float(point_list[0][1]), float(point_list[0][2])
							if self.minRange < z < self.maxRange:
								points.append([x, y, z])
								points_used.append((u, center_v))
					if len(points) > 0:
						points_np = np.array(points)
						zs = points_np[:, 2]
						percentile_90 = np.percentile(zs, 90)
						mask = zs >= percentile_90
						filtered_points = points_np[mask]
						filtered_points_used = [pt for idx, pt in enumerate(points_used) if mask[idx]]
						if len(filtered_points) > 0:
							raw_centroid = np.mean(filtered_points, axis=0)
							points_used = filtered_points_used
						else:
							raw_centroid = None
					else:
						raw_centroid = None
				
				# Mode 2: Only center pixel
				if self.centroid_mode == 2:
					center_u = int((x1 + x2) / 2)
					center_v = int((y1 + y2) / 2)

					p_dbg = pc2.read_points(pointcloud, field_names=['x', 'y', 'z'], skip_nans=False)
					point_gen = [p_dbg[int(center_v * pointcloud.width + center_u)]]
					
					for point in point_gen:
						if len(point) >= 3:
							x, y, z = float(point[0]), float(point[1]), float(point[2])
							if True: #self.minRange < z < self.maxRange:
								raw_centroid = np.array([x, y, z])
								points_used.append((center_u, center_v))
								break
					else:
						raw_centroid = None
					

					# ##############################################################################################
					# ###################################  DEBUG PURPOSE ONLY ######################################
					# ##############################################################################################
					# self.get_logger().info(f"{pointcloud.height}, {pointcloud.width}\n{self.latest_image.shape}")
					


					# p_dbg = pc2.read_points(pointcloud, field_names=['x', 'y', 'z'], skip_nans=False)
					# self.get_logger().info(f"{pointcloud.fields[0]}")
					
					# # Publish all points of p_dbg to /decoded_pcl
					# try:
					#     header = Header()
					#     header.stamp = self.get_clock().now().to_msg()
					#     header.frame_id = pointcloud.header.frame_id

					#     # Read all points from p_dbg generatorf
					#     all_points = [point for point in p_dbg if len(point) >= 3]
					#     if all_points:
					#         points_np = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in all_points], dtype=np.float32)
					#         decoded_cloud = pc2.create_cloud_xyz32(header, points_np)
					#         if not hasattr(self, 'decoded_pcl_publisher'):
					#             self.decoded_pcl_publisher = self.create_publisher(PointCloud2, '/decoded_pcl', 10)
					#         self.decoded_pcl_publisher.publish(decoded_cloud)
					# except Exception as e:
					#     self.get_logger().error(f"Error publishing decoded pcl: {str(e)}")

						
					# ##############################################################################################
					# ##############################################################################################
					# ##############################################################################################
				
				# Mode 1: Centroid of points with similar color
				elif self.centroid_mode == 1 and class_id in self.class_colors and self.latest_image is not None:
					# Convert cone color from BGR to HSV
					cone_color_bgr = np.uint8([[self.class_colors[class_id]]])
					cone_color_hsv = cv2.cvtColor(cone_color_bgr, cv2.COLOR_BGR2HSV)[0][0]

					# Convert the image to HSV once for all pixels
					image_hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)

					for i in range(0, len(uvs), 2):
						u, v = uvs[i], uvs[i + 1]

						# Read single point at this UV coordinate
						p_dbg = pc2.read_points(pointcloud, field_names=['x', 'y', 'z'], skip_nans=False)
						point_gen = [p_dbg[int(v * pointcloud.width + u)]]
						
						point_list = list(point_gen)

						if point_list and len(point_list[0]) >= 3:
							x, y, z = float(point_list[0][0]), float(point_list[0][1]), float(point_list[0][2])
							if self.minRange < z < self.maxRange:
								pixel_hsv = image_hsv[v, u]
								# Compute HSV distance (handle hue wrap-around)
								dh = min(abs(int(pixel_hsv[0]) - int(cone_color_hsv[0])), 180 - abs(int(pixel_hsv[0]) - int(cone_color_hsv[0])))
								ds = abs(int(pixel_hsv[1]) - int(cone_color_hsv[1]))
								dv = abs(int(pixel_hsv[2]) - int(cone_color_hsv[2]))
								color_diff = np.sqrt(dh**2 + ds**2 + dv**2)

								if color_diff <= self.color_tolerance:
									points.append([x, y, z])
									points_used.append((u, v))

					if len(points) > 0:
						raw_centroid = np.mean(points, axis=0)
					else:
						raw_centroid = None
				
				# Mode 0: Classic centroid of all points in bbox
				else:
					# Process each UV pair individually to ensure proper matching
					for i in range(0, len(uvs), 2):
						u, v = uvs[i], uvs[i + 1]
						uv_pair = [u, v]
						
						# Read single point at this UV coordinate
						p_dbg = pc2.read_points(pointcloud, field_names=['x', 'y', 'z'], skip_nans=False)
						point_gen = [p_dbg[int(v * pointcloud.width + u)]]
						
						point_list = list(point_gen)
						
						if point_list and len(point_list[0]) >= 3:
							x, y, z = float(point_list[0][0]), float(point_list[0][1]), float(point_list[0][2])
							if self.minRange < z < self.maxRange:
								points.append([x, y, z])
								points_used.append((u, v))

					# Only keep points in the 95th percentile of distance (z)
					if len(points) > 0:
						points_np = np.array(points)
						zs = points_np[:, 2]
						percentile_95 = np.percentile(zs, 95)
						mask = zs >= percentile_95
						points = points_np[mask].tolist()
						points_used = [pt for idx, pt in enumerate(points_used) if mask[idx]]
					
					if len(points) > 0:
						raw_centroid = np.mean(points, axis=0)
					else:
						raw_centroid = None
				
				# Apply EMA filter to the computed centroid
				if raw_centroid is not None:
					filtered_centroid = self.apply_ema_filter(raw_centroid, class_id)
				else:
					filtered_centroid = None

				if points_used and points:
					# Find min/max in 3D points
					points_np = np.array(points)
					min_point = points_np.min(axis=0)
					max_point = points_np.max(axis=0)
					self.get_logger().info(
						f"Detection class {class_id}: "
						f"PointCloud used {len(points)} points, "
						f"min=({min_point[0]:.2f}, {min_point[1]:.2f}, {min_point[2]:.2f}), "
						f"max=({max_point[0]:.2f}, {max_point[1]:.2f}, {max_point[2]:.2f})"
					)
				
				return filtered_centroid, points_used
				
			else:
				return None, None
		
		except Exception as e:
			self.get_logger().error(f"Error computing centroid: {str(e)}")
			import traceback
			self.get_logger().error(f"Traceback: {traceback.format_exc()}")
			return None, None
	
	def image_callback(self, msg):
		"""Handle incoming image messages"""
		try:
			self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
			self.latest_image_msg = msg
			self.latest_image_stamp = msg.header.stamp
			
			# Trigger processing if conditions are met
			self.try_process()
		except Exception as e:
			self.get_logger().error(f"Image callback error: {str(e)}")
	
	def pointcloud_callback(self, msg):
		"""Handle incoming pointcloud messages"""
		self.latest_pointcloud = msg
		self.latest_pointcloud_stamp = msg.header.stamp
		
		# Trigger processing if conditions are met
		self.try_process()

	def lidar_callback(self, msg):
		"""Handle laserscan data"""
		self.lidar_msg = msg
	
	def try_process(self):
		"""Check if we should process and do so if conditions are met"""
		current_time = time.time()
		
		# Check if we have both data and enough time has passed
		if (self.latest_image is not None and 
			self.latest_pointcloud is not None and
			not self.processing_in_progress and
			(current_time - self.last_process_time) >= self.min_process_interval):
			
			# Check timestamp synchronization (within 50ms)
			if (self.latest_image_stamp is not None and 
				self.latest_pointcloud_stamp is not None):
				time_diff = abs((self.latest_image_stamp.sec + self.latest_image_stamp.nanosec * 1e-9) -
							   (self.latest_pointcloud_stamp.sec + self.latest_pointcloud_stamp.nanosec * 1e-9))
				if time_diff > 0.05:  # 50ms threshold
					return
			
			with self.processing_lock:
				self.processing_in_progress = True
				self.last_process_time = current_time
				self.process_detection()
				self.processing_in_progress = False
	
	def process_detection(self):
		"""Main processing pipeline - optimized version"""
		try:
			total_start = cv2.getTickCount()
			
			# Preprocess image
			input_data = self.preprocess_image(self.latest_image)
			
			# Run inference
			start_time = cv2.getTickCount()
			output_data = self.do_inference(input_data)
			end_time = cv2.getTickCount()
			inf_ms = (end_time - start_time) * 1000.0 / cv2.getTickFrequency()
			
			# Post-process detections
			detections = self.postprocess_detections(output_data, self.latest_image.shape)
			
			# Extract centroids and collect visualization points
			centroids = []
			centroids_for_publish = []
			class_ids_for_publish = []  # Color
			all_centroid_points = []  # For visualization
			distances = []  # For distance tracking
			
			start_centroid = cv2.getTickCount()
			for detection in detections:
				centroid, points_used = self.compute_bbox_centroid(
					self.latest_pointcloud, detection['bbox'], detection
				)
								
				centroids.append(centroid)
				all_centroid_points.append(points_used)
				
				if centroid is not None:
					centroids_for_publish.append(centroid)
					class_ids_for_publish.append(detection['class_id'])
					distance = np.linalg.norm(centroid)
					distances.append(distance)
				else:
					distances.append(None)

			# Filter out cones that are detected nearer than 10 cm apart
			filtered_centroids = []
			filtered_class_ids = []
			min_dist = 0.05  # 10 cm

			for i, c1 in enumerate(centroids_for_publish):
				too_close = False
				for c2 in filtered_centroids:
					if np.linalg.norm(np.array(c1) - np.array(c2)) < min_dist:
						too_close = True
						break
				if not too_close:
					filtered_centroids.append(c1)
					filtered_class_ids.append(class_ids_for_publish[i])

			centroids_for_publish = filtered_centroids
			class_ids_for_publish = filtered_class_ids
			
			end_centroid = cv2.getTickCount()
			centroid_ms = (end_centroid - start_centroid) * 1000.0 / cv2.getTickFrequency()
			
			# Draw bounding boxes, centroid points, and distance info on visualization image
			annotated_image = self.draw_boxes(self.latest_image, detections, all_centroid_points, centroids)
			
			# Save debug information if enabled
			if self.debug_mode:
				self.save_debug_detections(detections, centroids, distances, inf_ms, centroid_ms)
			
			# Publish centroids as PointCloud2
			if centroids_for_publish:
				self.publish_centroids(centroids_for_publish, class_ids_for_publish)
			
			# Publish visualization image with bounding boxes and centroid points
			bbox_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
			bbox_msg.header = self.latest_image_msg.header
			self.bbox_image_publisher.publish(bbox_msg)
			
			# Track performance
			total_end = cv2.getTickCount()
			total_ms = (total_end - total_start) * 1000.0 / cv2.getTickFrequency()
			self.performance_window.append(total_ms)
			
			# Log detection info
			if detections:
				avg_time = np.mean(self.performance_window) if self.performance_window else total_ms
				self.get_logger().info(
					f'Detected {len(detections)} cones, {len(centroids_for_publish)} centroids | '
					f'Times: inf={inf_ms:.1f}ms, cent={centroid_ms:.1f}ms, total={total_ms:.1f}ms (avg={avg_time:.1f}ms)'
				)
				
		except Exception as e:
			self.get_logger().error(f"Processing error: {str(e)}")
			import traceback
			self.get_logger().error(f"Traceback: {traceback.format_exc()}")

	
	def lidar_detection(self, possible_cones):
		"""
		header:
			stamp:
				sec: 1751553787
				nanosec: 42062777
			frame_id: lidar_onBoard_link

		angle_min: 0.0 (0)
		angle_max: 6.27... (2pi)
		angle_increment: 0.00196713930927217
		time_increment: 3.0927240004530177e-05
		scan_time: 0.09875067323446274
		range_min: 0.05000000074505806
		range_max: 30.0
		ranges:
		- 4.195000171661377
		...
		intesities:
		- 47.0
		...
		"""
		self.viewed_cones = []

		if not hasattr(self, 'lidar_msg') or self.lidar_msg is None:
			return

		angle_inc = self.lidar_msg.angle_increment
		threshold_coneLidar = 0.2  # [m]

		# Precompute all scan points in XY using numpy for efficiency
		ranges = np.array(self.lidar_msg.ranges)
		num_points = len(ranges)
		angles = self.lidar_msg.angle_min + np.arange(num_points) * angle_inc
		scan_xy = np.stack([ranges * np.cos(angles), ranges * np.sin(angles)], axis=1)

		possible_cones_xy = np.array([[cone[0], cone[1]] for cone in possible_cones])

		# For each possible cone, find lidar points within threshold
		for cone_xy in possible_cones_xy:
			# Compute distances to all scan points at once
			dists = np.linalg.norm(scan_xy - cone_xy, axis=1)
			mask = dists < threshold_coneLidar
			nearby_points = scan_xy[mask]

			if nearby_points.shape[0] > 1:
				# Average the lidar points and the cone position
				mean_xy = np.mean(np.vstack((nearby_points, cone_xy)), axis=0)
				self.viewed_cones.append(mean_xy)

				self.get_logger().info(
					f"Lidar match: cone at {cone_xy}, mean at {mean_xy}, distance={np.linalg.norm(mean_xy - cone_xy):.2f}m"
				)

	
	def publish_centroids(self, centroids, class_ids):
		"""Publish cone centroids as PointCloud2 with class info as intensity"""
		try:
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self.latest_pointcloud.header.frame_id
			
			# Create points array with proper structure
			points = []

			cones_arr = []
					
			for centroid, class_id in zip(centroids, class_ids):
				x, y, z = centroid
				cone = ConePosition()
				
				# Get RGB values based on class_id
				if class_id == 7:
					r, g, b = 0, 0, 255  # Blue
					cone.type = 1

				elif class_id == 2:
					r, g, b = 255, 255, 0  # Yellow
					cone.type = 0

				else:
					r, g, b = 128, 128, 128  # Gray for unknown
					cone.type = 255
				
				# Pack RGB into 32-bit float (following ROS convention)
				rgb_packed = (int(r) << 16) | (int(g) << 8) | int(b)
				rgb_float = struct.unpack('f', struct.pack('I', rgb_packed))[0]

				# Add point as tuple (required by create_cloud)
				points.append((x, y, z, rgb_float))
				
				# Output cone_position_xyz
				cone.position = Point(x=x, y=y, z=z)

				cones_arr.append(cone)
				
			
			# Define PointCloud2 fields
			fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
			]
			
			# Create and publish the cloud
			cloud = pc2.create_cloud(header, fields, points)
			self.cone_publisher.publish(cloud)

			cones_xyz_stamped = ConePositionArrayStamped()

			cones_xyz_stamped.header = header
			cones_xyz_stamped.positions = cones_arr
			self.cone_position_publisher.publish(cones_xyz_stamped)



			# check if the cones are viewed also by the lidar
			self.lidar_detection(points)

			# publish position corrected detection
			# self.viewed_cones (list of [x, y]) to ConePositionArrayStamped
			viewed_cones_msg = ConePositionArrayStamped()
			viewed_cones_msg.header = header
			viewed_cones_msg.positions = []
			for xy in self.viewed_cones:
				cone = ConePosition()
				cone.position = Point(x=float(xy[0]), y=float(xy[1]), z=0.0)
				# Optionally set cone.type if you can infer it, else leave default
				viewed_cones_msg.positions.append(cone)
			self.viewed_cone_publisher.publish(viewed_cones_msg)

			
			# Log the published cones with their colors
			color_counts = {}
			for class_id in class_ids:
				color_name = self.class_names.get(class_id, f"Class_{class_id}")
				color_counts[color_name] = color_counts.get(color_name, 0) + 1
			
			self.get_logger().info(
				f"Published {len(centroids)} cones: {color_counts}"
			)
			
		except Exception as e:
			self.get_logger().error(f"Error publishing centroids: {str(e)}")
			import traceback
			self.get_logger().error(f"Traceback: {traceback.format_exc()}")

def main(args=None):
	rclpy.init(args=args)
	
	try:
		node = ConeDetectorNode()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	except Exception as e:
		print(f"Error: {str(e)}")
	finally:
		if 'node' in locals():
			node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()