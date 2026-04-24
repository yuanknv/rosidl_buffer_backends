cmake_minimum_required(VERSION 3.8)
project(torch_buffer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(torch_buffer_backend_msgs REQUIRED)
find_package(rmw REQUIRED)
find_package(rcutils REQUIRED)

find_package(cuda_buffer QUIET)
find_package(libtorch_vendor QUIET)
find_package(Torch REQUIRED)

set(_torch_has_cuda FALSE)
if(TARGET torch_cuda)
  set(_torch_has_cuda TRUE)
  find_package(CUDAToolkit QUIET)
endif()

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  INTERFACE
    ${TORCH_LIBRARIES}
    rosidl_buffer::rosidl_buffer
    rmw::rmw
    rcutils::rcutils
    ${torch_buffer_backend_msgs_TARGETS}
)

if(_torch_has_cuda AND CUDAToolkit_FOUND)
  target_link_libraries(${PROJECT_NAME} INTERFACE CUDA::cudart)
  # torch_buffer compiles no CUDA device code (no .cu, no nvcc), so CUDA's
  # host_config.h GCC check is irrelevant for us but still #errors out when
  # the host GCC major exceeds the toolkit's supported range. Suppress it.
  target_compile_definitions(${PROJECT_NAME} INTERFACE
    __NV_NO_HOST_COMPILER_CHECK=1)
endif()

if(cuda_buffer_FOUND)
  target_link_libraries(${PROJECT_NAME} INTERFACE cuda_buffer::cuda_buffer)
endif()

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(${PROJECT_NAME})
ament_export_dependencies(libtorch_vendor rosidl_buffer torch_buffer_backend_msgs rmw rcutils)
if(_torch_has_cuda AND CUDAToolkit_FOUND)
  ament_export_dependencies(CUDAToolkit)
endif()
if(cuda_buffer_FOUND)
  ament_export_dependencies(cuda_buffer)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_torch_buffer
    test/test_torch_buffer.cpp
  )
  if(TARGET test_torch_buffer)
    target_link_libraries(test_torch_buffer
      ${PROJECT_NAME}
      ${TORCH_LIBRARIES}
      rosidl_buffer::rosidl_buffer
    )
  endif()
endif()

ament_package()
