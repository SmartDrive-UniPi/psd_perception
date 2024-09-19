#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tinyxml2.h>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// Structure to represent a Cone
struct Cone {
    double x;
    double y;
    double z;
    std::string color_str;
};

// ROS2 Node Class
class ConeFilterNode : public rclcpp::Node {
public:
    ConeFilterNode() : Node("cone_filter_node") {
        // Declare parameters
        this->declare_parameter<std::string>("sdf_file", "/home/ubuntu/psd_ws/src/psd_gazebo_sim/psd_gazebo_worlds/world/track.sdf");
        this->declare_parameter<std::string>("pose_topic", "/psd_vehicle_pose");
        this->declare_parameter<double>("max_distance", 10.0);  // [meters]
        this->declare_parameter<double>("max_angle", 60.0 * (M_PI / 180.0)); // [radians]

        // Get parameter values
        this->get_parameter("sdf_file", sdf_file_);
        this->get_parameter("pose_topic", pose_topic_);
        this->get_parameter("max_distance", max_distance_);
        this->get_parameter("max_angle", max_angle_);

        RCLCPP_INFO(this->get_logger(), "Parameters loaded");

        // Load cones from SDF
        loadConesFromSDF(sdf_file_);

        RCLCPP_INFO(this->get_logger(), "Cones loaded: %lu cones", cones_.size());

        // Subscribe to pose topic
        pose_subscription_ = this->create_subscription<geometry_msgs::msg::Pose>(
            pose_topic_, 10, 
            std::bind(&ConeFilterNode::poseCallback, this, std::placeholders::_1));

        // Create publishers
        viewed_cones_marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "viewed_cones_marker", 10);

        viewed_cones_rbc_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "viewed_cones_RBCcolor", 10);
    }

private:
    // Function to load cones from SDF file
    void loadConesFromSDF(const std::string& sdf_path) {
        tinyxml2::XMLDocument doc;
        tinyxml2::XMLError eResult = doc.LoadFile(sdf_path.c_str());
        if (eResult != tinyxml2::XML_SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Error loading SDF file: %s", sdf_path.c_str());
            return;
        }

        tinyxml2::XMLElement* root = doc.FirstChildElement("sdf");
        if (root == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "No <sdf> root element found.");
            return;
        }

        // Navigate to <world>
        tinyxml2::XMLElement* world = root->FirstChildElement("world");
        if (world == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "No <world> element found in SDF.");
            return;
        }

        // Navigate to <model name='cones'>
        tinyxml2::XMLElement* model = world->FirstChildElement("model");
        while (model != nullptr) {
            const char* modelName = model->Attribute("name");
            if (modelName != nullptr && std::string(modelName) == "cones") {
                break;
            }
            model = model->NextSiblingElement("model");
        }

        if (model == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "No <model name='cones'> found in the SDF file.");
            return;
        }

        // Iterate through each <link> within the model
        tinyxml2::XMLElement* link = model->FirstChildElement("link");
        while (link != nullptr) {
            Cone cone;

            // Extract <pose>
            tinyxml2::XMLElement* poseElem = link->FirstChildElement("pose");
            if (poseElem != nullptr && poseElem->GetText() != nullptr) {
                std::string poseStr = poseElem->GetText();
                if (!parsePose(poseStr, cone.x, cone.y, cone.z)) {
                    RCLCPP_WARN(this->get_logger(), "Error parsing pose for link: %s", 
                        link->Attribute("name") ? link->Attribute("name") : "unknown");
                    cone.x = cone.y = cone.z = 0.0;
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "No <pose> found for link: %s", 
                    link->Attribute("name") ? link->Attribute("name") : "unknown");
                cone.x = cone.y = cone.z = 0.0;
            }

            // Extract color from <visual>/<material>/<ambient>
            tinyxml2::XMLElement* visual = link->FirstChildElement("visual");
            if (visual != nullptr) {
                tinyxml2::XMLElement* material = visual->FirstChildElement("material");
                if (material != nullptr) {
                    tinyxml2::XMLElement* ambient = material->FirstChildElement("ambient");
                    if (ambient != nullptr && ambient->GetText() != nullptr) {
                        cone.color_str = parseColor(ambient->GetText());
                    } else {
                        cone.color_str = "unknown";
                        RCLCPP_WARN(this->get_logger(), "No <ambient> color found for link: %s", 
                            link->Attribute("name") ? link->Attribute("name") : "unknown");
                    }
                } else {
                    cone.color_str = "unknown";
                    RCLCPP_WARN(this->get_logger(), "No <material> found for visual of link: %s", 
                        link->Attribute("name") ? link->Attribute("name") : "unknown");
                }
            } else {
                cone.color_str = "unknown";
                RCLCPP_WARN(this->get_logger(), "No <visual> found for link: %s", 
                    link->Attribute("name") ? link->Attribute("name") : "unknown");
            }

            // Add the cone to the list
            cones_.push_back(cone);

            // Move to the next <link>
            link = link->NextSiblingElement("link");
        }
    }

    // Helper function to parse the pose string
    bool parsePose(const std::string& poseStr, double& x, double& y, double& z) const {
        std::istringstream iss(poseStr);
        if (!(iss >> x >> y >> z)) {
            return false;
        }
        return true;
    }

    // Helper function to parse the ambient color string
    std::string parseColor(const std::string& ambientStr) const {
        std::istringstream iss(ambientStr);
        double r, g, b, a;
        if (!(iss >> r >> g >> b >> a)) {
            return "unknown";
        }

        // Simple color detection based on RGB values
        if (r == 1.0 && g == 1.0 && b == 0.0) {
            return "yellow";
        } else if (r == 0.0 && g == 0.0 && b == 1.0) {
            return "blue";
        } else {
            std::ostringstream color_stream;
            color_stream << "unknown(" << r << ", " << g << ", " << b << ")";
            return color_stream.str();
        }
    }

    // Callback function for vehicle pose
    void poseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        // Yaw calculation from quaternion
        tf2::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
        tf2::Matrix3x3 m(q);

        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // Filter cones
        visualization_msgs::msg::MarkerArray viewed_markers;  
        std_msgs::msg::Float32MultiArray viewed_cones_rbc;

        int id = 0;  // Marker ID
        for (const auto& cone : cones_) {
            // Calculate distance and angle
            double dx = cone.x - msg->position.x;
            double dy = cone.y - msg->position.y;
            double distance = std::sqrt(dx * dx + dy * dy);
            double cone_yaw = std::atan2(dy, dx);  
            double angle = cone_yaw - yaw;

            // Normalize angle to range [-pi, pi]
            angle = std::fmod(angle + M_PI, 2 * M_PI);
            if (angle < 0)
                angle += 2 * M_PI;
            angle -= M_PI;

            if (distance <= max_distance_ && std::abs(angle) <= max_angle_) { 
                RCLCPP_DEBUG(this->get_logger(), "Viewed cone - Distance: %f, Angle: %f", distance, angle);

                // Create a marker for the viewed cone
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "home";
                marker.header.stamp = this->get_clock()->now();
                marker.ns = "viewed_cones_marker";
                marker.id = id++;
                marker.type = visualization_msgs::msg::Marker::SPHERE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.pose.position.x = cone.x;
                marker.pose.position.y = cone.y;
                marker.pose.position.z = cone.z;
                marker.pose.orientation.w = 1.0;  // No rotation
                marker.scale.x = 0.5;   // Adjust size
                marker.scale.y = 0.5;
                marker.scale.z = 0.5;
                marker.color.a = 0.8;  // Alpha (transparency)

                // Convert color_str to RGB
                if (cone.color_str == "yellow") { // Yellow
                    marker.color.r = 1.0;
                    marker.color.g = 1.0;
                    marker.color.b = 0.0;
                }
                else if (cone.color_str == "blue") { // Blue
                    marker.color.r = 0.0;
                    marker.color.g = 0.0;
                    marker.color.b = 1.0;
                }
                else { // Unknown color
                    marker.color.r = 0.5;
                    marker.color.g = 0.5;
                    marker.color.b = 0.5;
                }

                viewed_markers.markers.push_back(marker);

                // Add viewed cone with range, bearing, and color
                viewed_cones_rbc.data.push_back(static_cast<float>(distance));       // Range
                viewed_cones_rbc.data.push_back(static_cast<float>(angle));          // Bearing
                viewed_cones_rbc.data.push_back(static_cast<float>(colorStrToNumeric(cone.color_str))); // Color
            }
        }

        // Publish viewed markers
        viewed_cones_marker_publisher_->publish(viewed_markers);
        viewed_cones_rbc_publisher_->publish(viewed_cones_rbc);
    }

    // Helper function to convert color string to numerical value
    double colorStrToNumeric(const std::string& color_str) const {
        if (color_str == "yellow") {
            return 1.0;
        } else if (color_str == "blue") {
            return 2.0;
        } else {
            RCLCPP_WARN(this->get_logger(), "Unknown color '%s'. Defaulting to 0.0.", color_str.c_str());
            return 0.0;
        }
    }

    // Member variables
    std::vector<Cone> cones_;

    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viewed_cones_marker_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr viewed_cones_rbc_publisher_;

    std::string sdf_file_;
    std::string pose_topic_;
    double max_distance_;
    double max_angle_; 
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConeFilterNode>());
    rclcpp::shutdown();
    return 0;
}
