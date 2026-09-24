<!--
Copyright 2026 Open Source Robotics Foundation, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# cuda_core_vendor

Installs `cuda-core` 0.3.1 and its Cargo.lock dependency set as a Cargo local
registry, preserving upstream archives, licenses, and checksums. Requires Python 3.12+ and Rust 1.89+ to build; compiling the CUDA
crates also requires CUDA 13+ and libclang.

Install a locally built Debian package:

```bash
sudo apt install ./ros-rolling-cuda-core-vendor_*.deb
source /opt/ros/rolling/setup.bash
```

Or build from source:

```bash
colcon build --packages-select cuda_core_vendor
source install/setup.bash
```

Declare `cuda_core_vendor` in a Rust ROS package's `package.xml` and use
`cuda-core = "=0.3.1"` in its `Cargo.toml`. Colcon discovers the CUDA crates in
the ament index. Other Cargo dependencies follow the application's configuration.

For offline builds of the pinned dependency set, pass
`--config <prefix>/share/cuda_core_vendor/.cargo/config.toml` to Cargo.
The first vendor build downloads the locked crates. For an offline vendor build,
set `CARGO_NET_OFFLINE=true` and pass
`--cmake-args -DCUDA_CORE_VENDOR_CACHE=<prefix>/share/cuda_core_vendor/registry`
to colcon.
