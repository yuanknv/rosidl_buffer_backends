//! Receive CUDA or CPU images through the input-buffer adapter.
use std::sync::Arc;

use cuda_buffer_rs::from_input_buffer;
use cuda_core::{CudaContext, CudaStream};
use rclrs::{Context, CreateBasicExecutor, RclrsErrorFilter, SpinOptions, SubscriptionOptions};
use ros_env::sensor_msgs::msg::buffer::Image;

fn print_image(image: Image, stream: &Arc<CudaStream>) -> Result<(), Box<dyn std::error::Error>> {
    let expected = u64::from(image.height) * u64::from(image.step);
    if image.data.is_empty() || image.data.len() as u64 != expected {
        return Err("image payload is empty or does not match height * step".into());
    }
    let backend = image.data.backend_name()?;
    let input = from_input_buffer::<u8>(&image.data, stream)?;
    let pixels = input.to_host_vec()?;
    println!("received backend={backend} pixels={pixels:?}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let mut executor = Context::default_from_env()?.create_basic_executor();
    let node = executor.create_node("cuda_image_subscriber")?;
    let _subscription = node.create_subscription::<Image, _>(
        SubscriptionOptions::new("image").acceptable_buffer_backends("cuda"),
        move |image: Image| {
            if let Err(error) = print_image(image, &stream) {
                eprintln!("receive failed: {error}");
            }
        },
    )?;
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
