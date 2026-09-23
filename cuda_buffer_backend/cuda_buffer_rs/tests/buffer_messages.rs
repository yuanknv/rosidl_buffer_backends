use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rclrs::{Context, CreateBasicExecutor, SpinOptions};
use ros_env::{rcl_interfaces, sensor_msgs, test_msgs};
use rosidl_runtime_rs::{Buffer, Message};

fn make_buffer(backend: &str, values: &[u8]) -> Buffer<u8> {
    let mut buffer = Buffer::from(vec![0u8; values.len()]);
    if backend == "cuda" {
        let stream = cuda_core::CudaContext::new(0).unwrap().default_stream();
        cuda_buffer_rs::from_output_buffer::<u8>(&mut buffer, &stream)
            .unwrap()
            .copy_from_host(values)
            .unwrap();
    } else {
        buffer.as_mut_slice().unwrap().copy_from_slice(values);
    }
    buffer
}

#[test]
fn cuda_storage_materializes_recursively_without_changing_the_cpu_api() {
    let mut nested = test_msgs::msg::buffer::MultiNested::default();
    let unbounded = test_msgs::msg::buffer::UnboundedSequences {
        uint8_values: make_buffer("cuda", &[9, 7, 5]),
        ..Default::default()
    };
    nested.array_of_unbounded_sequences[0] = unbounded.clone();
    nested.bounded_sequence_of_unbounded_sequences = vec![unbounded.clone()].try_into().unwrap();
    nested.unbounded_sequence_of_unbounded_sequences = vec![unbounded];
    nested.array_of_bounded_sequences[0].uint8_values =
        make_buffer("cuda", &[3, 2, 1]).try_into().unwrap();
    let native =
        test_msgs::msg::buffer::MultiNested::into_rmw_message(Cow::Borrowed(&nested)).into_owned();
    assert!(native.array_of_unbounded_sequences[0]
        .uint8_values
        .is_rosidl_buffer());
    let cpu = test_msgs::msg::MultiNested::try_from_rmw_message(native).unwrap();
    assert_eq!(
        cpu.array_of_unbounded_sequences[0].uint8_values,
        vec![9, 7, 5]
    );
    assert_eq!(
        cpu.bounded_sequence_of_unbounded_sequences[0]
            .uint8_values
            .as_slice(),
        &[9, 7, 5]
    );
    assert_eq!(
        cpu.unbounded_sequence_of_unbounded_sequences[0].uint8_values,
        vec![9, 7, 5]
    );
    assert_eq!(
        cpu.array_of_bounded_sequences[0].uint8_values.as_slice(),
        &[3, 2, 1]
    );
    assert_eq!(nested.try_into_cpu().unwrap(), cpu);
}

#[test]
fn owned_transport_conversion_retains_the_gpu_allocation() {
    let image = sensor_msgs::msg::buffer::Image {
        data: make_buffer("cuda", &[4, 8, 12]),
        ..Default::default()
    };
    let pointer = image.data.as_sequence().rosidl_buffer_ptr();
    assert!(pointer.is_some());
    assert!(image.data.as_slice().is_none());
    let native = sensor_msgs::msg::buffer::Image::into_rmw_message(Cow::Owned(image)).into_owned();
    assert_eq!(native.data.rosidl_buffer_ptr(), pointer);
    let image = sensor_msgs::msg::buffer::Image::from_rmw_message(native);
    assert_eq!(image.data.as_sequence().rosidl_buffer_ptr(), pointer);
    assert_eq!(image.data.to_vec().unwrap(), vec![4, 8, 12]);
    let copy = image.clone();
    drop(image);
    assert_eq!(copy.try_into_cpu().unwrap().data, vec![4, 8, 12]);
}

#[test]
fn cpu_and_buffer_json_have_the_same_schema() {
    let image = sensor_msgs::msg::Image {
        data: vec![1, 2, 3],
        ..Default::default()
    };
    let json = serde_json::to_string(&image).unwrap();
    let portable: sensor_msgs::msg::buffer::Image = serde_json::from_str(&json).unwrap();
    assert_eq!(serde_json::to_string(&portable).unwrap(), json);
    {
        let gpu = sensor_msgs::msg::buffer::Image {
            data: make_buffer("cuda", &[1, 2, 3]),
            ..Default::default()
        };
        assert_eq!(serde_json::to_string(&gpu).unwrap(), json);
    }
}

fn deliver_once(backend: &str) {
    let mut executor = Context::default().create_basic_executor();
    let node = executor.create_node("portable_image_test").unwrap();
    let topic = format!("portable_image_{}_{}", backend, std::process::id());
    let received = Arc::new(Mutex::new((None, None)));
    let cpu_received = Arc::clone(&received);
    let _cpu = node
        .create_subscription::<sensor_msgs::msg::Image, _>(
            &topic,
            move |image: sensor_msgs::msg::Image| {
                assert_eq!(image.width, 3);
                cpu_received.lock().unwrap().0 = Some(image.data);
            },
        )
        .unwrap();
    let buffer_received = Arc::clone(&received);
    let _portable = node
        .create_subscription::<sensor_msgs::msg::buffer::Image, _>(
            rclrs::SubscriptionOptions::new(&topic).acceptable_buffer_backends(backend),
            move |image: sensor_msgs::msg::buffer::Image| {
                let name = image.data.backend_name().unwrap();
                buffer_received.lock().unwrap().1 = Some((name, image.data.to_vec().unwrap()));
            },
        )
        .unwrap();
    let publisher = node
        .create_publisher::<sensor_msgs::msg::buffer::Image>(&topic)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while publisher.get_subscription_count().unwrap() < 2 {
        assert!(Instant::now() < deadline, "subscriber discovery timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
    let image = sensor_msgs::msg::buffer::Image {
        width: 3,
        data: make_buffer(backend, &[11, 22, 33]),
        ..Default::default()
    };
    publisher.publish(image).unwrap();
    loop {
        let errors = executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(20)));
        assert!(
            errors.iter().all(rclrs::RclrsError::is_timeout),
            "{errors:?}"
        );
        let results = received.lock().unwrap();
        if let (Some(cpu), Some((name, portable))) = &*results {
            assert_eq!(cpu, &[11, 22, 33]);
            assert_eq!(portable, cpu);
            assert_eq!(name, backend);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "one publication did not reach both representations: {results:?}"
        );
    }
}

#[test]
fn one_cuda_publication_reaches_both_representations() {
    deliver_once("cuda");
}

fn service_roundtrip(backend: &'static str) {
    use rcl_interfaces::{msg, srv};
    let mut executor = Context::default().create_basic_executor();
    let node = executor.create_node("portable_service_test").unwrap();
    let name = format!("portable_service_{}_{}", backend, std::process::id());
    let _service = node
        .create_service::<srv::buffer::GetParameters, _>(
            &name,
            move |request: srv::buffer::GetParameters_Request| {
                assert_eq!(request.names, vec!["pixels"]);
                srv::buffer::GetParameters_Response {
                    values: vec![msg::buffer::ParameterValue {
                        byte_array_value: make_buffer(backend, &[13, 17, 23]),
                        ..Default::default()
                    }],
                }
            },
        )
        .unwrap();
    let client = node.create_client::<srv::GetParameters>(&name).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !client.service_is_ready().unwrap() {
        assert!(Instant::now() < deadline, "service discovery timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
    let received = Arc::new(Mutex::new(None));
    let output = Arc::clone(&received);
    let _call = client
        .call_then(
            srv::GetParameters_Request {
                names: vec!["pixels".into()],
            },
            move |response: srv::GetParameters_Response| {
                *output.lock().unwrap() = Some(response.values[0].byte_array_value.clone());
            },
        )
        .unwrap();
    loop {
        let errors = executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(20)));
        assert!(
            errors.iter().all(rclrs::RclrsError::is_timeout),
            "{errors:?}"
        );
        if let Some(values) = &*received.lock().unwrap() {
            assert_eq!(values, &[13, 17, 23]);
            break;
        }
        assert!(Instant::now() < deadline, "service response timed out");
    }
}

#[test]
fn cpu_client_receives_nested_cuda_service_response() {
    service_roundtrip("cuda");
}

#[test]
fn buffer_client_sends_nested_cuda_request_to_cpu_service() {
    use rcl_interfaces::{msg, srv};
    let mut executor = Context::default().create_basic_executor();
    let node = executor.create_node("portable_request_test").unwrap();
    let name = format!("portable_request_{}", std::process::id());
    let _service = node
        .create_service::<srv::SetParameters, _>(&name, |request: srv::SetParameters_Request| {
            assert_eq!(request.parameters[0].value.byte_array_value, vec![29, 31]);
            srv::SetParameters_Response {
                results: vec![msg::SetParametersResult {
                    successful: true,
                    reason: "received".into(),
                }],
            }
        })
        .unwrap();
    let client = node
        .create_client::<srv::buffer::SetParameters>(&name)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !client.service_is_ready().unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(10));
    }
    let request = srv::buffer::SetParameters_Request {
        parameters: vec![msg::buffer::Parameter {
            name: "pixels".into(),
            value: msg::buffer::ParameterValue {
                byte_array_value: make_buffer("cuda", &[29, 31]),
                ..Default::default()
            },
        }],
    };
    let received = Arc::new(Mutex::new(false));
    let output = Arc::clone(&received);
    let _call = client
        .call_then(
            request,
            move |response: srv::buffer::SetParameters_Response| {
                assert!(response.results[0].successful);
                assert_eq!(response.results[0].reason, "received");
                *output.lock().unwrap() = true;
            },
        )
        .unwrap();
    while !*received.lock().unwrap() {
        assert!(Instant::now() < deadline, "service response timed out");
        let errors = executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(20)));
        assert!(
            errors.iter().all(rclrs::RclrsError::is_timeout),
            "{errors:?}"
        );
    }
}
