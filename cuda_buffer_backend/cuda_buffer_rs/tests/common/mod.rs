// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

mod audit;

use std::ffi::c_void;
use std::process::{Child, Command};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use cuda_buffer_rs::{allocate_buffer, from_input_buffer, from_output_buffer, to_buffer, CopyKind};
use cuda_core::{launch_kernel_on_stream, CudaContext, CudaFunction, CudaStream};
use rclrs::{
    Context, CreateBasicExecutor, Executor, Node, Publisher, SpinOptions, Subscription,
    SubscriptionOptions,
};
use ros_env::{
    sensor_msgs::msg::{self, buffer::Image},
    std_msgs::msg::UInt32,
};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;
const BYTES: usize = (WIDTH * HEIGHT) as usize;
const SAMPLES: u32 = 10;
const CHILD: &str = "CUDA_BUFFER_TEST_SUBSCRIBER";
const FILL: &str = r#"
.version 7.0
.target sm_52
.address_size 64
.visible .entry fill(.param .u64 address, .param .u32 size, .param .u32 seed) {
    .reg .u32 i, block, threads, n, v;
    .reg .u64 p, offset;
    .reg .pred done;
    mov.u32 i, %tid.x;
    mov.u32 block, %ctaid.x;
    mov.u32 threads, %ntid.x;
    mad.lo.u32 i, block, threads, i;
    ld.param.u32 n, [size];
    setp.ge.u32 done, i, n;
    @done bra end;
    ld.param.u64 p, [address];
    ld.param.u32 v, [seed];
    mul.lo.u32 v, v, 31;
    mad.lo.u32 v, i, 17, v;
    cvt.u64.u32 offset, i;
    add.u64 p, p, offset;
    st.global.u8 [p], v;
end:
    ret;
}
"#;

fn spin(executor: &mut Executor, deadline: Instant) {
    assert!(Instant::now() < deadline, "image delivery timed out");
    let errors = executor.spin(SpinOptions::spin_once().timeout(Duration::from_millis(10)));
    assert!(
        errors.iter().all(rclrs::RclrsError::is_timeout),
        "{errors:?}"
    );
}

fn verify_pixels(sequence: u32, pixels: &[u8]) {
    assert_eq!(pixels.len(), BYTES);
    for (index, &value) in pixels.iter().enumerate() {
        assert_eq!(
            value,
            (index as u32 * 17 + sequence * 31) as u8,
            "pixel {index}"
        );
    }
}

fn verify_copy_counter(stream: &Arc<CudaStream>) {
    let before = audit::copied_bytes();
    let mut buffer = allocate_buffer(4).unwrap();
    let mut output = from_output_buffer::<u8>(&mut buffer, stream).unwrap();
    let values = [1u8, 2, 3, 4];
    assert!(!audit::is_imported(output.get_ptr(), 4));
    // SAFETY: values remains live through completion of this control copy.
    unsafe {
        to_buffer(
            values.as_ptr().cast(),
            4,
            &mut output,
            stream,
            CopyKind::HostToDevice,
        )
    }
    .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        audit::copied_bytes(),
        before + 4,
        "CUDA copy hook is inactive"
    );
    assert_eq!(buffer.to_vec().unwrap(), values);
    assert_eq!(audit::copied_bytes(), before + 8);
}

fn publish_image(
    publisher: &Publisher<Image>,
    sequence: u32,
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
) {
    let mut image = Image {
        height: HEIGHT,
        width: WIDTH,
        encoding: "mono8".into(),
        step: WIDTH,
        data: allocate_buffer(BYTES).unwrap(),
        ..Default::default()
    };
    image.header.stamp.sec = sequence as i32;
    let mut output = from_output_buffer::<u8>(&mut image.data, stream).unwrap();
    let mut address = unsafe { output.as_device_buffer_mut() }.cu_deviceptr();
    let mut size = BYTES as u32;
    let mut seed = sequence;
    let mut args = [
        (&mut address as *mut u64).cast::<c_void>(),
        (&mut size as *mut u32).cast(),
        (&mut seed as *mut u32).cast(),
    ];
    stream
        .launch_host_function(|| std::thread::sleep(Duration::from_millis(30)))
        .unwrap();
    // SAFETY: the kernel writes exactly BYTES bytes; the caller retains its module.
    unsafe {
        launch_kernel_on_stream(
            function,
            (BYTES as u32 / 256, 1, 1),
            (256, 1, 1),
            0,
            stream,
            &mut args,
        )
    }
    .unwrap();
    publisher.publish(image).unwrap();
}

fn gpu_subscriber(
    node: &Node,
    topic: &str,
    id: usize,
    strict_copies: bool,
) -> (Subscription<Image>, Arc<AtomicU32>) {
    let context = CudaContext::new(0).unwrap();
    let streams = [context.new_stream().unwrap(), context.default_stream()];
    verify_copy_counter(&streams[0]);
    let ack = node
        .create_publisher::<UInt32>(&*format!("{topic}_ack"))
        .unwrap();
    let received = Arc::new(AtomicU32::new(0));
    let count = received.clone();
    let baseline = Arc::new(Mutex::new(None));
    let subscription = node
        .create_subscription::<Image, _>(
            SubscriptionOptions::new(topic).acceptable_buffer_backends("cuda"),
            move |message: Image| {
                let sequence = message.header.stamp.sec as u32;
                if sequence == 0
                    && (message.data.is_empty() || message.data.backend_name().unwrap() != "cuda")
                {
                    return;
                }
                assert_eq!(message.data.backend_name().unwrap(), "cuda");
                assert_eq!(
                    (message.width, message.height, message.step),
                    (WIDTH, HEIGHT, WIDTH)
                );
                assert_eq!(message.encoding, "mono8");
                let before = audit::copied_bytes();
                if sequence > 0 && strict_copies {
                    assert_eq!(
                        Some(before),
                        *baseline.lock().unwrap(),
                        "subscriber copied the payload"
                    );
                }
                let stream = &streams[(sequence > SAMPLES / 2) as usize];
                let input = from_input_buffer::<u8>(&message.data, stream).unwrap();
                assert!(
                    audit::is_imported(input.get_ptr(), BYTES),
                    "received data is not an imported VMM mapping"
                );
                assert_eq!(input.stream().cu_stream(), stream.cu_stream());
                assert_eq!(
                    audit::copied_bytes(),
                    before,
                    "input adapter copied the payload"
                );
                // The host copy validates pixels after the transport assertions.
                verify_pixels(sequence, &input.to_host_vec().unwrap());
                if sequence == 0 {
                    *baseline.lock().unwrap() = Some(audit::copied_bytes());
                } else {
                    assert_eq!(
                        sequence,
                        count.fetch_add(1, Ordering::SeqCst) + 1,
                        "duplicate or out-of-order image"
                    );
                }
                ack.publish(UInt32 {
                    data: ((id as u32) << 16) | sequence,
                })
                .unwrap();
            },
        )
        .unwrap();
    (subscription, received)
}

struct Children(Vec<Child>);
impl Drop for Children {
    fn drop(&mut self) {
        for child in &mut self.0 {
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

pub fn run(test_name: &str, separate_processes: bool, gpu_count: usize, with_cpu: bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut executor = Context::default().create_basic_executor();
    let node = executor
        .create_node(&*format!("cuda_images_{}", std::process::id()))
        .unwrap();
    if let Ok(role) = std::env::var(CHILD) {
        let (id, topic) = role.split_once(':').unwrap();
        let (_subscription, count) = gpu_subscriber(&node, topic, id.parse().unwrap(), true);
        while count.load(Ordering::SeqCst) < SAMPLES {
            spin(&mut executor, deadline);
        }
        return;
    }

    let topic = format!("{test_name}_{}", std::process::id());
    let total = gpu_count + usize::from(with_cpu);
    let acks = Arc::new(Mutex::new(vec![u32::MAX; total]));
    let progress = acks.clone();
    let _ack = node
        .create_subscription::<UInt32, _>(&*format!("{topic}_ack"), move |ack: UInt32| {
            let id = (ack.data >> 16) as usize;
            let sequence = ack.data & 0xffff;
            let mut counts = progress.lock().unwrap();
            assert!(id < counts.len());
            if sequence > 0 {
                assert_eq!(sequence, counts[id] + 1);
            }
            counts[id] = sequence;
        })
        .unwrap();
    let publisher = node.create_publisher::<Image>(&topic).unwrap();
    let mut subscriptions = Vec::new();
    let mut children = Children(Vec::new());
    for id in 0..gpu_count {
        if separate_processes {
            children.0.push(
                Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", test_name, "--nocapture"])
                    .env(CHILD, format!("{id}:{topic}"))
                    .spawn()
                    .unwrap(),
            );
        } else {
            subscriptions.push(gpu_subscriber(&node, &topic, id, !with_cpu));
        }
    }
    let _cpu = with_cpu.then(|| {
        let ack = node
            .create_publisher::<UInt32>(&*format!("{topic}_ack"))
            .unwrap();
        node.create_subscription::<msg::Image, _>(&topic, move |message: msg::Image| {
            let sequence = message.header.stamp.sec as u32;
            verify_pixels(sequence, &message.data);
            ack.publish(UInt32 {
                data: ((gpu_count as u32) << 16) | sequence,
            })
            .unwrap();
        })
        .unwrap()
    });
    let context = CudaContext::new(0).unwrap();
    let streams = [context.default_stream(), context.new_stream().unwrap()];
    verify_copy_counter(&streams[0]);
    let module = context.load_module_from_ptx_src(FILL).unwrap();
    let function = module.load_function("fill").unwrap();
    while publisher.get_subscription_count().unwrap() < total {
        spin(&mut executor, deadline);
    }
    // Warm up descriptor negotiation before measuring ten individual publications.
    while acks.lock().unwrap().contains(&u32::MAX) {
        publish_image(&publisher, 0, &streams[0], &function);
        for _ in 0..10 {
            spin(&mut executor, deadline);
        }
    }
    let before = audit::copied_bytes();
    for sequence in 1..=SAMPLES {
        let stream = &streams[(sequence > SAMPLES / 2) as usize];
        publish_image(&publisher, sequence, stream, &function);
        while acks.lock().unwrap().iter().any(|&n| n != sequence) {
            for child in &mut children.0 {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "subscriber exited: {status}");
                }
            }
            spin(&mut executor, deadline);
        }
    }
    if with_cpu {
        assert!(
            audit::copied_bytes() >= before + SAMPLES as usize * BYTES,
            "CPU fallback did not read the CUDA payload"
        );
    } else {
        assert_eq!(
            audit::copied_bytes(),
            before,
            "CUDA publication copied the payload"
        );
    }
    for child in &mut children.0 {
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            spin(&mut executor, deadline);
        }
    }
    context.check_err().unwrap();
}
