<!--
Copyright 2026 Open Source Robotics Foundation, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# cuda_buffer_rs

Rust access to ROS CUDA buffers for publishing and consuming GPU data.

## Build from source

Requires Linux, Rust 1.89+, CUDA 13+, and ROS 2 with `colcon-cargo`,
`colcon-ros-cargo`, and `cargo-ament-build`.
[`cuda_core_vendor`](../../cuda_core_vendor/README.md) supplies the pinned
`cuda-core` 0.3.1 dependency.

From a ROS workspace containing the matching Buffer branches and message packages:

```bash
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --packages-up-to cuda_buffer_rs
source install/setup.bash
```

In your application's `Cargo.toml`:

```toml
[dependencies]
cuda_buffer_rs = { version = "0.1", features = ["cuda-core"] }
cuda-core = "=0.3.1"
rclrs = "0.7"
ros-env = "=0.2.0"
```

Declare `cuda_buffer_rs`, `rclrs`, and `sensor_msgs` as dependencies in
`package.xml`, then build the application with `colcon build`.

## Usage

These snippets use an existing ROS `node`.
`produce_device_data` and `consume_device_data` represent your CUDA kernels,
queued on that stream.

### Publisher (direct write, zero-copy)

```rust,ignore
use cuda_buffer_rs::{allocate_buffer, from_output_buffer};
use cuda_core::CudaContext;
use ros_env::sensor_msgs::msg::buffer::Image;

let context = CudaContext::new(0)?;
let stream = context.new_stream()?;
let publisher = node.create_publisher::<Image>("image")?;
let mut image = Image {
    height: 480,
    width: 640,
    encoding: "rgb8".into(),
    step: 640 * 3,
    data: allocate_buffer(640 * 480 * 3)?,
    ..Default::default()
};
let mut output = from_output_buffer::<u8>(&mut image.data, &stream)?;
unsafe {
    produce_device_data(output.get_ptr(), output.len(), &stream);
}
publisher.publish(image)?;
```

Queue a kernel that writes the entire buffer on `stream`.

### Publisher (copy from an existing pointer)

With `image.data` already allocated, copy from an existing device allocation:

```rust,ignore
use cuda_buffer_rs::{from_output_buffer, to_buffer, CopyKind};

let mut output = from_output_buffer::<u8>(&mut image.data, &stream)?;
unsafe {
    to_buffer(
        gpu_ptr,
        byte_count,
        &mut output,
        &stream,
        CopyKind::DeviceToDevice,
    )?;
}
publisher.publish(image)?;
```

Use `CopyKind::HostToDevice` for a host pointer. Keep the source allocation valid
until the copy completes, and order its producer before the copy.

### Subscriber (read from a buffer, zero-copy)

```rust,ignore
use cuda_buffer_rs::from_input_buffer;
use cuda_core::CudaContext;
use rclrs::SubscriptionOptions;
use ros_env::sensor_msgs::msg::buffer::Image;

let context = CudaContext::new(0)?;
let stream = context.new_stream()?;
let subscription = node.create_subscription::<Image, _>(
    SubscriptionOptions::new("image").acceptable_buffer_backends("cuda"),
    move |image: Image| {
        let input = from_input_buffer::<u8>(&image.data, &stream).unwrap();
        unsafe {
            consume_device_data(input.get_ptr(), input.len(), &stream);
        }
    },
)?;
```

`input.get_ptr()` points directly to the received CUDA storage. CPU input is
uploaded by `from_input_buffer`.

CPU applications can keep `sensor_msgs::msg::Image` and its `Vec<u8>` payload.
For CPU-backed `msg::buffer::Image`, use `data.as_slice()` to borrow the pixels.
