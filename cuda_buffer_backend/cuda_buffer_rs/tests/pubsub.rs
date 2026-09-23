use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use cuda_buffer_rs::{allocate_buffer, from_input_buffer, from_output_buffer};
use cuda_core::CudaContext;
use rclrs::{Context, CreateBasicExecutor, SpinOptions, SubscriptionOptions};
use ros_env::std_msgs::msg::buffer::UInt8MultiArray;

const SUBSCRIBER_TOPIC: &str = "CUDA_BUFFER_RS_SUBSCRIBER_TOPIC";

struct Subscriber(Child);
impl Drop for Subscriber {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

fn payload() -> Vec<u32> {
    (0..1024).map(|i| i * 17 + 3).collect()
}

#[test]
fn cuda_buffer_survives_rust_pubsub() {
    if let Ok(topic) = std::env::var(SUBSCRIBER_TOPIC) {
        run_subscriber(&topic);
        return;
    }

    let cuda = CudaContext::new(0).expect("CUDA primary context");
    let stream = cuda.default_stream();
    let expected = payload();
    let mut publisher_executor = Context::default().create_basic_executor();
    let publisher_node_name = format!("cuda_buffer_rs_publisher_{}", std::process::id());
    let publisher_node = publisher_executor
        .create_node(&*publisher_node_name)
        .expect("create publisher node");
    let topic = format!("cuda_buffer_rs_topic_{}", std::process::id());
    let publisher = publisher_node
        .create_publisher::<UInt8MultiArray>(&topic)
        .expect("create publisher");
    let mut subscriber = Subscriber(
        Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", "cuda_buffer_survives_rust_pubsub", "--nocapture"])
            .env(SUBSCRIBER_TOPIC, &topic)
            .spawn()
            .expect("start subscriber"),
    );

    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let mut message = UInt8MultiArray {
            data: allocate_buffer(expected.len() * 4).unwrap(),
            ..Default::default()
        };
        let mut output =
            from_output_buffer::<u32>(&mut message.data, &stream).expect("typed writer");
        output
            .copy_from_host(&expected)
            .expect("initialize payload");
        publisher.publish(message).expect("publish CUDA buffer");
        let errors =
            publisher_executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(10)));
        assert!(
            errors.iter().all(rclrs::RclrsError::is_timeout),
            "publisher executor: {errors:?}"
        );
        if let Some(status) = subscriber.0.try_wait().expect("query subscriber") {
            assert!(status.success(), "subscriber failed: {status}");
            return;
        }
        thread::sleep(Duration::from_millis(50));
        assert!(Instant::now() < deadline, "CUDA message was not received");
    }
}

fn run_subscriber(topic: &str) {
    let cuda = CudaContext::new(0).expect("CUDA primary context");
    let stream = cuda.new_stream().expect("nonblocking consumer stream");
    let mut executor = Context::default().create_basic_executor();
    let node = executor
        .create_node("cuda_buffer_rs_subscriber")
        .expect("create subscriber node");
    let received = Arc::new(AtomicBool::new(false));
    let callback_received = Arc::clone(&received);
    let _subscription = node
        .create_subscription::<UInt8MultiArray, _>(
            SubscriptionOptions::new(topic).acceptable_buffer_backends("cuda"),
            move |message: UInt8MultiArray| {
                if message.data.is_empty() {
                    println!("CUDA_IPC_EMPTY_DESCRIPTOR_DROPPED");
                    return;
                }
                let Some(raw) = message.data.as_sequence().rosidl_buffer_ptr() else {
                    return;
                };
                // SAFETY: message.data retains this native Buffer through the callback.
                if !unsafe { cuda_buffer_rs::is_cuda_backed(raw) } {
                    return;
                }
                let handle = from_input_buffer::<u32>(&message.data, &stream)
                    .expect("acquire imported CUDA buffer");
                assert_eq!(
                    handle.to_host_vec().expect("read imported CUDA data"),
                    payload()
                );
                println!("CUDA_IPC_PAYLOAD_OK elements={}", handle.len());
                callback_received.store(true, Ordering::Release);
            },
        )
        .expect("create CUDA subscription");

    let deadline = Instant::now() + Duration::from_secs(15);
    while !received.load(Ordering::Acquire) {
        let errors = executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(100)));
        assert!(
            errors.iter().all(rclrs::RclrsError::is_timeout),
            "subscriber executor: {errors:?}"
        );
        assert!(Instant::now() < deadline, "CUDA message was not received");
    }
}
