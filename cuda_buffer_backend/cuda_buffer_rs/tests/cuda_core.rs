use std::ffi::c_void;
use std::sync::Arc;
use std::time::Duration;

use cuda_buffer_rs::{
    allocate_buffer, from_input_buffer, from_output_buffer, get_primitive_sequence_read_handle,
    CudaBuffer, ErrorKind,
};
use cuda_core::{launch_kernel_on_stream, CudaContext, CudaStream, DeviceBuffer};

const FILL_PTX: &str = r#"
.version 7.0
.target sm_52
.address_size 64
.visible .entry fill(.param .u64 address, .param .u32 value) {
    .reg .u32 i, v;
    .reg .u64 p, offset;
    ld.param.u64 p, [address];
    ld.param.u32 v, [value];
    mov.u32 i, %tid.x;
    mul.wide.u32 offset, i, 4;
    add.u64 p, p, offset;
    st.global.u32 [p], v;
    ret;
}
"#;

// Launches a 32-element fill kernel on the facade's device address.
fn fill(buffer: &mut DeviceBuffer<u32>, stream: &CudaStream, value: u32) {
    let module = stream.context().load_module_from_ptx_src(FILL_PTX).unwrap();
    let function = module.load_function("fill").unwrap();
    let mut address = buffer.cu_deviceptr();
    let mut value = value;
    let mut arguments = [
        (&mut address as *mut u64).cast::<c_void>(),
        (&mut value as *mut u32).cast::<c_void>(),
    ];
    // Keep the loaded module alive until this helper synchronizes the stream.
    unsafe { launch_kernel_on_stream(&function, (1, 1, 1), (32, 1, 1), 0, stream, &mut arguments) }
        .unwrap();
    stream.synchronize().unwrap();
}

fn stream_pair(default_writer: bool, default_reader: bool) {
    let context = CudaContext::new(0).unwrap();
    // Create both streams before queueing delayed work (new_stream may sync).
    let producer = if default_writer {
        context.default_stream()
    } else {
        context.new_stream().unwrap()
    };
    let consumer = if default_reader {
        context.default_stream()
    } else {
        context.new_stream().unwrap()
    };
    let module = context.load_module_from_ptx_src(FILL_PTX).unwrap();
    let function = module.load_function("fill").unwrap();
    let mut buffer = allocate_buffer(32 * 4).unwrap();
    let mut writer = from_output_buffer::<u32>(&mut buffer, &producer).unwrap();
    writer.copy_from_host(&[0; 32]).unwrap();
    let address = writer.device_ptr() as usize;
    println!(
        "VMM_ADDRESS=0x{address:x} WRITER_STREAM={:p} READER_STREAM={:p}",
        producer.cu_stream(),
        consumer.cu_stream()
    );
    assert_eq!(writer.len(), 32);
    assert_eq!(writer.byte_len(), 128);
    assert_eq!(writer.stream().cu_stream().is_null(), default_writer);
    let mut address_arg = unsafe { writer.as_device_buffer_mut() }.cu_deviceptr();
    assert_eq!(address_arg as usize, address);
    let mut value = 0xa5a5_1234u32;
    let mut arguments = [
        (&mut address_arg as *mut u64).cast::<c_void>(),
        (&mut value as *mut u32).cast::<c_void>(),
    ];
    producer
        .launch_host_function(|| std::thread::sleep(Duration::from_millis(100)))
        .unwrap();
    unsafe {
        launch_kernel_on_stream(
            &function,
            (1, 1, 1),
            (32, 1, 1),
            0,
            &producer,
            &mut arguments,
        )
    }
    .unwrap();
    drop(writer);
    // No producer synchronization: the backend read wait must order the kernel.
    let reader = from_input_buffer::<u32>(&buffer, &consumer).unwrap();
    assert_eq!(reader.device_ptr() as usize, address);
    assert_eq!(
        unsafe { reader.as_device_buffer() }.cu_deviceptr() as usize,
        address
    );
    let actual = reader.to_host_vec().unwrap();
    // Always drain the producer before asserting, even if ordering regresses.
    producer.synchronize().unwrap();
    assert_eq!(actual, vec![value; 32]);
    drop(reader);
    drop(buffer);
    context
        .check_err()
        .expect("facade must not call cuMemFree on VMM storage");
}

#[test]
fn default_stream_writer_to_nonblocking_reader() {
    stream_pair(true, false);
}

#[test]
fn nonblocking_writer_to_default_stream_reader() {
    stream_pair(false, true);
}

#[test]
fn two_distinct_nonblocking_streams() {
    stream_pair(false, false);
}

#[test]
fn both_default_stream_handles() {
    stream_pair(true, true);
}

#[test]
fn typed_facade_kernel_and_message_ownership_roundtrip() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    // Audit control: this ordinary allocation must produce a cuMemFree call.
    let ordinary = DeviceBuffer::from_host(&stream, &[1u32; 32]).unwrap();
    println!("ORDINARY_ADDRESS=0x{:x}", ordinary.cu_deviceptr());
    drop(ordinary);
    let baseline = Arc::strong_count(&stream);
    let mut buffer = CudaBuffer::allocate(128).unwrap();
    let address;
    {
        let mut writer = buffer.get_write_handle::<u32>(&stream).unwrap();
        assert_eq!(Arc::strong_count(&stream), baseline + 1);
        address = writer.device_ptr() as usize;
        println!("VMM_ADDRESS=0x{address:x}");
        // Same stream, no replacement/move of the facade, immediate submission.
        fill(unsafe { writer.as_device_buffer_mut() }, &stream, 47);
    }
    assert_eq!(Arc::strong_count(&stream), baseline);
    let sequence = buffer.into_primitive_sequence();
    let handle = get_primitive_sequence_read_handle::<u32>(&sequence, &stream).unwrap();
    assert_eq!(handle.device_ptr() as usize, address);
    assert_eq!(handle.to_host_vec().unwrap(), vec![47; 32]);
    drop(handle);
    let reclaimed = CudaBuffer::from_primitive_sequence(sequence).unwrap();
    assert_eq!(
        reclaimed
            .get_read_handle::<u32>(&stream)
            .unwrap()
            .to_host_vec()
            .unwrap(),
        vec![47; 32]
    );
    context.check_err().unwrap();
}

#[test]
fn invalid_layout_is_rejected_before_write_acquisition() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let mut buffer = CudaBuffer::allocate(7).unwrap();
    assert_eq!(
        buffer.get_write_handle::<u32>(&stream).err().unwrap().kind,
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        buffer.get_write_handle::<()>(&stream).err().unwrap().kind,
        ErrorKind::InvalidArgument
    );
    let mut writer = buffer.get_write_handle::<u8>(&stream).unwrap();
    assert_eq!(
        writer.copy_from_host(&[1, 2]).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    writer.copy_from_host(&[3; 7]).unwrap();
    drop(writer);
    assert_eq!(
        buffer
            .get_read_handle::<u8>(&stream)
            .unwrap()
            .to_host_vec()
            .unwrap(),
        vec![3; 7]
    );
    assert!(CudaBuffer::allocate(0)
        .unwrap()
        .get_read_handle::<u8>(&stream)
        .is_err());
}

#[test]
fn stream_and_context_references_are_released_on_drop_and_unwind() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let context_refs = Arc::strong_count(&context);
    let stream_refs = Arc::strong_count(&stream);
    let mut buffer = CudaBuffer::allocate(128).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut writer = buffer.get_write_handle::<u32>(&stream).unwrap();
        writer.copy_from_host(&[91; 32]).unwrap();
        panic!("exercise unwinding after native acquisition");
    }));
    assert!(result.is_err());
    assert_eq!(Arc::strong_count(&context), context_refs);
    assert_eq!(Arc::strong_count(&stream), stream_refs);
    for _ in 0..64 {
        let handle = buffer.get_read_handle::<u32>(&stream).unwrap();
        assert_eq!(handle.to_host_vec().unwrap(), vec![91; 32]);
    }
    assert_eq!(Arc::strong_count(&context), context_refs);
    assert_eq!(Arc::strong_count(&stream), stream_refs);
    context.check_err().unwrap();
}

#[test]
fn handle_retains_stream_when_caller_drops_its_arc() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let weak = Arc::downgrade(&stream);
    let mut buffer = CudaBuffer::allocate(128).unwrap();
    let mut writer = buffer.get_write_handle::<u32>(&stream).unwrap();
    drop(stream);
    assert!(weak.upgrade().is_some());
    writer.copy_from_host(&[12; 32]).unwrap();
    drop(writer);
    assert!(weak.upgrade().is_none());
    let stream = context.default_stream();
    assert_eq!(
        buffer
            .get_read_handle::<u32>(&stream)
            .unwrap()
            .to_host_vec()
            .unwrap(),
        vec![12; 32]
    );
}

#[test]
fn finalized_typed_write_allows_multiple_readers() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut buffer = CudaBuffer::allocate(16).unwrap();
    let mut writer = buffer.get_write_handle::<u32>(&stream).unwrap();
    assert!(!writer.is_empty());
    assert_eq!(writer.device_ptr_mut().cast_const(), writer.device_ptr());
    assert!(format!("{writer:?}").contains("len: 4"));
    writer.copy_from_host(&[1, 2, 3, 4]).unwrap();
    drop(writer);
    assert_eq!(
        buffer.get_write_handle::<u32>(&stream).unwrap_err().kind,
        ErrorKind::Cuda
    );
    let first = buffer.get_read_handle::<u32>(&stream).unwrap();
    let second = buffer.get_read_handle::<u32>(&stream).unwrap();
    assert!(!first.is_empty());
    assert_eq!(first.device_ptr(), second.device_ptr());
    assert!(format!("{first:?}").contains("len: 4"));
    drop(first);
    assert_eq!(second.to_host_vec().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn invalid_read_layout_leaves_valid_typed_reads_available() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let mut buffer = CudaBuffer::allocate(7).unwrap();
    buffer
        .get_write_handle::<u8>(&stream)
        .unwrap()
        .copy_from_host(&[6; 7])
        .unwrap();
    assert_eq!(
        buffer.get_read_handle::<u32>(&stream).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        buffer.get_read_handle::<()>(&stream).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        buffer
            .get_read_handle::<u8>(&stream)
            .unwrap()
            .to_host_vec()
            .unwrap(),
        vec![6; 7]
    );
}

#[test]
fn output_promotion_accepts_kernel_results_without_a_host_upload() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut data = rosidl_runtime_rs::Buffer::from(vec![0u8; 128]);
    let address;
    {
        let mut output = from_output_buffer::<u32>(&mut data, &stream).unwrap();
        address = output.device_ptr() as usize;
        println!("VMM_ADDRESS=0x{address:x}");
        // SAFETY: fill only writes contents on the handle's acquisition stream.
        fill(unsafe { output.as_device_buffer_mut() }, &stream, 73);
    }
    assert_eq!(data.backend_name().unwrap(), "cuda");
    let input = from_input_buffer::<u32>(&data, &stream).unwrap();
    assert_eq!(input.device_ptr() as usize, address);
    assert_eq!(input.to_host_vec().unwrap(), vec![73; 32]);
    drop(input);
    drop(data);
    context.check_err().unwrap();
}
