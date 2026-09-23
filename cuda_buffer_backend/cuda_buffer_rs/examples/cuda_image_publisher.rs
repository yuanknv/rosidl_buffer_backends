// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Publish CUDA-backed images through the output-buffer adapter.
use std::sync::Arc;
use std::time::Duration;

use cuda_buffer_rs::{allocate_buffer, from_output_buffer, to_buffer, CopyKind};
use cuda_core::{CudaContext, CudaStream};
use rclrs::{Context, CreateBasicExecutor, Publisher, RclrsErrorFilter, SpinOptions};
use ros_env::sensor_msgs::msg::buffer::Image;

static PIXELS: [u8; 3] = [10, 20, 30];

fn publish_image(
    publisher: &Publisher<Image>,
    stream: &Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut image = Image {
        height: 1,
        width: 3,
        encoding: "mono8".into(),
        step: 3,
        data: allocate_buffer(3)?,
        ..Default::default()
    };
    let mut output = from_output_buffer::<u8>(&mut image.data, stream)?;
    // SAFETY: PIXELS is immutable static storage, valid through copy completion.
    unsafe {
        to_buffer(
            PIXELS.as_ptr().cast(),
            PIXELS.len(),
            &mut output,
            stream,
            CopyKind::HostToDevice,
        )?;
    }
    publisher.publish(image)?;
    println!("published CUDA image");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let mut executor = Context::default_from_env()?.create_basic_executor();
    let node = executor.create_node("cuda_image_publisher")?;
    let publisher = node.create_publisher::<Image>("image")?;
    let _timer = node.create_timer_repeating(Duration::from_secs(1), move || {
        if let Err(error) = publish_image(&publisher, &stream) {
            eprintln!("publish failed: {error}");
        }
    })?;
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
