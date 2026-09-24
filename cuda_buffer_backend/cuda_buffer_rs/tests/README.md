<!--
Copyright 2026 Open Source Robotics Foundation, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Tests

Use a CUDA-capable GPU and the Rust colcon plugins listed in the package README.
From the built ROS workspace:

```bash
source install/setup.bash
RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=0 \
  colcon test --packages-select cuda_buffer_rs --return-code-on-test-failure
colcon test-result --verbose
```

`colcon.pkg` enables `cuda-core` for ROS builds and tests. The suite includes
unit tests, same-process and separate-process CUDA delivery, CPU fallback,
message/service compatibility, and compile-fail doctests in `compile_fail_doctests/mod.rs`.
The pub/sub tests validate pixel data, CUDA backend selection, image metadata,
message order, and access on default and non-blocking CUDA streams.
