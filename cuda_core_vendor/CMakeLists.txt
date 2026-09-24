# Copyright 2026 Open Source Robotics Foundation, Inc.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(cuda_core_vendor NONE)

find_package(ament_cmake REQUIRED)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(CUDA_CORE_VENDOR_CACHE "" CACHE PATH "Pre-fetched Cargo local registry")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  tests/Cargo.lock tests/Cargo.toml cmake/vendor.py)

set(cache_args)
if(CUDA_CORE_VENDOR_CACHE)
  list(APPEND cache_args --cache "${CUDA_CORE_VENDOR_CACHE}")
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendor.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/Cargo.lock"
    "${CMAKE_CURRENT_BINARY_DIR}" ${cache_args}
  COMMAND_ERROR_IS_FATAL ANY
)

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/registry"
  DESTINATION "share/${PROJECT_NAME}")
install(DIRECTORY .cargo tests DESTINATION "share/${PROJECT_NAME}")
install(FILES LICENSE README.md DESTINATION "share/${PROJECT_NAME}")

# colcon-ros-cargo resolves these crates through the ament index.
foreach(crate cuda-core-0.3.1 cuda-bindings-0.3.1 cuda-core-derive-0.3.1 oxide-artifacts-0.2.1)
  string(REGEX REPLACE "-[0-9]+[.][0-9]+[.][0-9]+$" "" name "${crate}")
  install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sources/${crate}/"
    DESTINATION "share/${name}/rust")
  ament_index_register_resource("rust_packages" PACKAGE_NAME "${name}")
endforeach()

if(BUILD_TESTING)
  enable_testing()
  configure_file(.cargo/config.toml .cargo/config.toml COPYONLY)
  add_test(NAME offline_compile
    COMMAND "${CMAKE_COMMAND}" -E env
      "CARGO_HOME=${CMAKE_CURRENT_BINARY_DIR}/test-cargo-home"
      "${CARGO_EXECUTABLE}" check --offline --locked
      --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/tests/Cargo.toml"
      --target-dir "${CMAKE_CURRENT_BINARY_DIR}/test-target"
      --config "${CMAKE_CURRENT_BINARY_DIR}/.cargo/config.toml")
  set_tests_properties(offline_compile PROPERTIES TIMEOUT 300)
endif()

ament_package()
