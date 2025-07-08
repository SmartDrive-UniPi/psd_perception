The current code is the `camera_node.py` that if not used is stored as `camera_node_debug.py`. \
\
This can be launched using this command: \
`ros2 run psd_perception camera_node --ros-args     -p engine_path:=/home/psd/psd_ws/src/psd_perception/best.engine     -p conf_threshold:=0.3     -p input_size:=640 -p debug:=true -p debug_csv_path:=debug_3.csv`