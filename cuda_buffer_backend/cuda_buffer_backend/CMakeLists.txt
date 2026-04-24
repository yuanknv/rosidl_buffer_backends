cmake_minimum_required(VERSION 3.8)
project(cuda_buffer_backend)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer_backend REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(rosidl_buffer_backend_registry REQUIRED)
find_package(pluginlib REQUIRED)
find_package(cuda_buffer REQUIRED)
find_package(cuda_buffer_backend_msgs REQUIRED)
find_package(rcutils REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/cuda_buffer_backend_plugin.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${cuda_buffer_backend_msgs_TARGETS}
  rosidl_buffer_backend::rosidl_buffer_backend
  rosidl_runtime_cpp::rosidl_runtime_cpp
  rosidl_buffer_backend_registry::rosidl_buffer_backend_registry
  rosidl_typesupport_cpp::rosidl_typesupport_cpp
  pluginlib::pluginlib
  cuda_buffer::cuda_buffer
  rcutils::rcutils
  CUDA::cudart
  CUDA::cuda_driver
)

# The plugin is dlopen'd by pluginlib at runtime; its install RPATH needs the
# CUDA toolkit library directory so libcudart.so.<abi> can be resolved without
# relying on the caller's LD_LIBRARY_PATH.
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

install(
  FILES cuda_buffer_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(rosidl_buffer_backend cuda_buffer_plugin.xml)

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rosidl_buffer_backend
  rosidl_runtime_cpp
  rosidl_buffer_backend_registry
  rosidl_typesupport_cpp
  pluginlib
  cuda_buffer
  cuda_buffer_backend_msgs
  rcutils
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_add_gtest(test_cuda_buffer_backend
    test/test_cuda_buffer_backend.cpp
  )
  if(TARGET test_cuda_buffer_backend)
    target_link_libraries(test_cuda_buffer_backend
      ${PROJECT_NAME}
      rosidl_buffer_backend::rosidl_buffer_backend
      rosidl_runtime_cpp::rosidl_runtime_cpp
      rosidl_buffer_backend_registry::rosidl_buffer_backend_registry
      ${cuda_buffer_backend_msgs_TARGETS}
      CUDA::cudart
    )
  endif()

  add_library(cuda_image_publisher_component SHARED
    test/src/cuda_image_publisher_node.cpp
  )
  target_link_libraries(cuda_image_publisher_component
    cuda_buffer::cuda_buffer
    CUDA::cudart
    rclcpp::rclcpp
    rclcpp_components::component
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    rosidl_runtime_cpp::rosidl_runtime_cpp
  )
  rclcpp_components_register_node(cuda_image_publisher_component
    PLUGIN "CudaImagePublisher"
    EXECUTABLE cuda_image_publisher_node
  )
  install(TARGETS cuda_image_publisher_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  add_library(cuda_image_subscriber_component SHARED
    test/src/cuda_image_subscriber_node.cpp
  )
  target_link_libraries(cuda_image_subscriber_component
    cuda_buffer::cuda_buffer
    CUDA::cudart
    rclcpp::rclcpp
    rclcpp_components::component
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    rosidl_runtime_cpp::rosidl_runtime_cpp
  )
  rclcpp_components_register_node(cuda_image_subscriber_component
    PLUGIN "CudaImageSubscriber"
    EXECUTABLE cuda_image_subscriber_node
  )
  install(TARGETS cuda_image_subscriber_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  add_library(cuda_image_gpu_relay_component SHARED
    test/src/cuda_image_gpu_relay_node.cpp
  )
  target_link_libraries(cuda_image_gpu_relay_component
    cuda_buffer::cuda_buffer
    CUDA::cudart
    rclcpp::rclcpp
    rclcpp_components::component
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    rosidl_runtime_cpp::rosidl_runtime_cpp
  )
  rclcpp_components_register_node(cuda_image_gpu_relay_component
    PLUGIN "CudaImageGpuRelay"
    EXECUTABLE cuda_image_gpu_relay_node
  )
  install(TARGETS cuda_image_gpu_relay_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  add_launch_test(test/test_cuda_image_inter_pubsub_fastrtps_launch.py
    TARGET test_cuda_image_inter_pubsub_fastrtps_launch
    TIMEOUT 30
  )

  add_launch_test(test/test_cuda_image_intra_pubsub_fastrtps_launch.py
    TARGET test_cuda_image_intra_pubsub_fastrtps_launch
    TIMEOUT 30
  )

  add_launch_test(test/test_cuda_image_mixed_pubsub_fastrtps_launch.py
    TARGET test_cuda_image_mixed_pubsub_fastrtps_launch
    TIMEOUT 30
  )

  add_launch_test(test/test_cuda_image_multiprocess_fastrtps_launch.py
    TARGET test_cuda_image_multiprocess_fastrtps_launch
    TIMEOUT 45
  )

  add_launch_test(test/test_cuda_image_cpu_fallback_fastrtps_launch.py
    TARGET test_cuda_image_cpu_fallback_fastrtps_launch
    TIMEOUT 60
  )

  add_launch_test(test/test_cuda_image_mixed_no_duplicate_fastrtps_launch.py
    TARGET test_cuda_image_mixed_no_duplicate_fastrtps_launch
    TIMEOUT 45
  )

  add_launch_test(test/test_cuda_image_multi_pub_fastrtps_launch.py
    TARGET test_cuda_image_multi_pub_fastrtps_launch
    TIMEOUT 45
  )

  add_launch_test(test/test_cuda_image_gpu_pipeline_fastrtps_launch.py
    TARGET test_cuda_image_gpu_pipeline_fastrtps_launch
    TIMEOUT 45
  )
endif()

ament_package()
