cmake_minimum_required(VERSION 3.8)
project(psd_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(tinyxml2 REQUIRED)  # TinyXML-2

# Declare your node as an executable
add_executable(fake_perception src/fake_perception.cpp)  # Adjust if using a different name

# Specify dependencies in ament_target_dependencies
ament_target_dependencies(fake_perception
  rclcpp
  geometry_msgs
  visualization_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  custom_msgs
)

# Link TinyXML-2 library
target_link_libraries(fake_perception
  tinyxml2::tinyxml2
)

# Install the node
install(TARGETS fake_perception
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # The following lines skip specific linters
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
