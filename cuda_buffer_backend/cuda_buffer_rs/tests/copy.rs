// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;
use std::mem::size_of_val;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_buffer_rs::{
    allocate_buffer, from_input_buffer, from_output_buffer, to_buffer, CopyKind, ErrorKind,
};
use cuda_core::{CudaContext, DeviceBuffer};

#[test]
fn allocation_returns_message_storage_and_handles_expose_pointer_types() {
    let stream = CudaContext::new(0).unwrap().default_stream();
    let mut data: rosidl_runtime_rs::Buffer<u8> = allocate_buffer(4).unwrap();
    assert_eq!(data.backend_name().unwrap(), "cuda");
    let owner = data.as_sequence().rosidl_buffer_ptr();
    let pointer: *mut u8;
    {
        let mut output = from_output_buffer::<u8>(&mut data, &stream).unwrap();
        pointer = output.get_ptr();
        assert_eq!(pointer, output.device_ptr_mut());
        output.copy_from_host(&[1, 2, 3, 4]).unwrap();
    }
    let input = from_input_buffer::<u8>(&data, &stream).unwrap();
    let readable: *const u8 = input.get_ptr();
    assert_eq!(readable, pointer.cast_const());
    assert_eq!(input.to_host_vec().unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(data.as_sequence().rosidl_buffer_ptr(), owner);
    assert!(allocate_buffer(0).unwrap().is_empty());
}

#[test]
fn host_copy_uses_bytes_on_default_and_nonblocking_streams() {
    let context = CudaContext::new(0).unwrap();
    for stream in [context.default_stream(), context.new_stream().unwrap()] {
        let values = [3u32, 7, 11, 19];
        let mut data = allocate_buffer(size_of_val(&values)).unwrap();
        {
            let mut output = from_output_buffer::<u32>(&mut data, &stream).unwrap();
            // SAFETY: values stays live and unchanged through stream synchronization.
            unsafe {
                to_buffer(
                    values.as_ptr().cast(),
                    size_of_val(&values),
                    &mut output,
                    &stream,
                    CopyKind::HostToDevice,
                )
            }
            .unwrap();
            stream.synchronize().unwrap();
        }
        let input = from_input_buffer::<u32>(&data, &stream).unwrap();
        assert_eq!(input.to_host_vec().unwrap(), values);
    }
}

#[test]
fn device_copy_is_async_and_preserves_the_output_allocation() {
    let context = CudaContext::new(0).unwrap();
    let producer = context.new_stream().unwrap();
    let consumer = context.default_stream();
    let values = [23u32; 32];
    let source = DeviceBuffer::from_host(&producer, &values).unwrap();
    let mut data = allocate_buffer(size_of_val(&values)).unwrap();
    let owner = data.as_sequence().rosidl_buffer_ptr();
    let mut output = from_output_buffer::<u32>(&mut data, &producer).unwrap();
    let gate = Arc::new(AtomicBool::new(false));
    let callback_gate = Arc::clone(&gate);
    producer
        .launch_host_function(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            while !callback_gate.load(Ordering::Acquire) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            callback_gate.store(true, Ordering::Release);
        })
        .unwrap();
    // SAFETY: source and its completed contents remain live through the consumer read.
    let copied = unsafe {
        to_buffer(
            source.cu_deviceptr() as usize as *const c_void,
            size_of_val(&values),
            &mut output,
            &producer,
            CopyKind::DeviceToDevice,
        )
    };
    let returned_before_gate_opened = !gate.load(Ordering::Acquire);
    copied.unwrap();
    let input = from_input_buffer::<u32>(&data, &consumer).unwrap();
    gate.store(true, Ordering::Release);
    assert_eq!(input.to_host_vec().unwrap(), values);
    assert!(
        returned_before_gate_opened,
        "copy synchronized the producer stream"
    );
    assert_eq!(data.as_sequence().rosidl_buffer_ptr(), owner);
    context.check_err().unwrap();
}

#[test]
fn invalid_copy_requests_leave_the_output_usable() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other = context.new_stream().unwrap();
    let mut data = allocate_buffer(4).unwrap();
    let values = [29u8; 4];
    {
        let mut output = from_output_buffer::<u8>(&mut data, &stream).unwrap();
        for (source, size, selected) in [
            (std::ptr::null(), 4, &stream),
            (values.as_ptr().cast(), 5, &stream),
            (values.as_ptr().cast(), 4, &other),
        ] {
            // SAFETY: invalid arguments must be rejected before submitting a copy.
            let error =
                unsafe { to_buffer(source, size, &mut output, selected, CopyKind::HostToDevice) }
                    .unwrap_err();
            assert_eq!(error.kind, ErrorKind::InvalidArgument);
        }
        // SAFETY: zero bytes access no source memory.
        unsafe {
            to_buffer(
                std::ptr::null(),
                0,
                &mut output,
                &other,
                CopyKind::HostToDevice,
            )
        }
        .unwrap();
        output.copy_from_host(&values).unwrap();
    }
    assert_eq!(
        from_input_buffer::<u8>(&data, &stream)
            .unwrap()
            .to_host_vec()
            .unwrap(),
        values
    );
}

#[test]
fn unpublished_owner_can_drop_before_unused_write_handle() {
    static VALUES: [u8; 4] = [3, 5, 7, 11];
    let context = CudaContext::new(0).unwrap();
    for stream in [context.default_stream(), context.new_stream().unwrap()] {
        let mut data = allocate_buffer(VALUES.len()).unwrap();
        let mut output = from_output_buffer::<u8>(&mut data, &stream).unwrap();
        // SAFETY: VALUES remains valid through the queued copy.
        unsafe {
            to_buffer(
                VALUES.as_ptr().cast(),
                VALUES.len(),
                &mut output,
                &stream,
                CopyKind::HostToDevice,
            )
        }
        .unwrap();
        drop(data);
        stream.synchronize().unwrap();
    }
    context.check_err().unwrap();
}
