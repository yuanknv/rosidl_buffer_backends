// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use cuda_buffer_rs::{get_primitive_sequence_read_handle, CudaBuffer, CudaStream, ErrorKind};
use cuda_core::CudaContext;
use ros_env::std_msgs::msg::rmw::UInt8MultiArray;
use rosidl_runtime_rs::PrimitiveSequence;

extern "C" {
    fn rosidl_buffer_uint8_create_cpu(
        data: *const u8,
        size: usize,
        buffer: *mut *mut c_void,
    ) -> i32;
}

fn cpu_buffer(values: &[u8]) -> CudaBuffer {
    let mut raw = std::ptr::null_mut();
    assert_eq!(
        unsafe { rosidl_buffer_uint8_create_cpu(values.as_ptr(), values.len(), &mut raw) },
        0
    );
    // SAFETY: create_cpu returned a uniquely owned Buffer with this byte length.
    unsafe { CudaBuffer::from_raw(raw, values.len()) }.unwrap()
}

#[test]
fn cpu_storage_is_rejected_by_typed_access_and_promoted_by_raw_access() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let mut buffer = cpu_buffer(&[1, 2, 3, 4]);
    assert_eq!(
        buffer.get_read_handle::<u8>(&stream).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        buffer.get_write_handle::<u8>(&stream).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    let sequence = buffer.into_primitive_sequence();
    assert_eq!(
        get_primitive_sequence_read_handle::<u8>(&sequence, &stream)
            .unwrap_err()
            .kind,
        ErrorKind::InvalidArgument
    );
    let sequence = CudaBuffer::from_primitive_sequence(sequence).unwrap_err();
    {
        let read =
            cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).unwrap();
        assert_eq!(read.len(), 4);
        assert!(!read.device_ptr().is_null());
    }
    let raw = sequence.into_owned_rosidl_buffer().unwrap();
    // SAFETY: the sequence transferred its sole native owner.
    let mut buffer = unsafe { CudaBuffer::from_raw(raw, 4) }.unwrap();
    {
        let write = buffer.write(CudaStream::INTERNAL).unwrap();
        assert_eq!(write.len(), 4);
    }
    assert!(buffer.is_cuda_backed());
}

#[test]
fn normal_sequences_reject_typed_cuda_reads() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let sequence = PrimitiveSequence::from(&[1u8, 2, 3, 4][..]);
    assert_eq!(
        get_primitive_sequence_read_handle::<u8>(&sequence, &stream)
            .unwrap_err()
            .kind,
        ErrorKind::InvalidArgument
    );
}

#[test]
fn cloned_messages_keep_cuda_storage_and_compare_contents() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut buffer = CudaBuffer::allocate(4).unwrap();
    buffer
        .get_write_handle::<u8>(&stream)
        .unwrap()
        .copy_from_host(&[3, 5, 7, 9])
        .unwrap();
    let message = UInt8MultiArray {
        data: buffer.into_primitive_sequence(),
        ..Default::default()
    };
    assert!(format!("{message:?}").contains("PrimitiveSequence"));
    let mut cloned = message.clone();
    let pointer = cloned.data.rosidl_buffer_ptr();
    assert!(std::panic::catch_unwind(|| cloned.data.as_slice()).is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cloned.data.as_mut_slice();
    }))
    .is_err());
    assert_eq!(cloned.data.rosidl_buffer_ptr(), pointer);
    assert!(cloned.data.is_rosidl_buffer());
    assert_eq!(message, cloned);
    assert_eq!(cloned.data, PrimitiveSequence::from(&[3u8, 5, 7, 9][..]));
    assert_ne!(cloned.data, PrimitiveSequence::from(&[0u8; 4][..]));
    drop(message);
    let read = get_primitive_sequence_read_handle::<u8>(&cloned.data, &stream).unwrap();
    assert_eq!(read.to_host_vec().unwrap(), vec![3, 5, 7, 9]);
}

#[test]
fn nonowning_message_storage_is_returned_on_rejected_adoption() {
    #[repr(C)]
    struct NativeSequence {
        data: *mut c_void,
        size: usize,
        capacity: usize,
        is_buffer: bool,
        owns_buffer: bool,
    }
    let owner = CudaBuffer::allocate(4).unwrap();
    let native = NativeSequence {
        data: owner.as_ptr(),
        size: 4,
        capacity: 4,
        is_buffer: true,
        owns_buffer: false,
    };
    // SAFETY: the native sequence layout matches PrimitiveSequence<u8>. The
    // owner remains live and owns_buffer=false prevents duplicate destruction.
    let sequence: PrimitiveSequence<u8> = unsafe { std::mem::transmute(native) };
    let sequence = CudaBuffer::from_primitive_sequence(sequence).unwrap_err();
    assert_eq!(sequence.rosidl_buffer_ptr(), Some(owner.as_ptr()));
    drop(sequence);
    assert!(owner.is_cuda_backed());
}

#[test]
fn message_owner_can_move_to_a_thread_with_its_cuda_context() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let mut buffer = CudaBuffer::allocate(16).unwrap();
    buffer
        .get_write_handle::<u32>(&stream)
        .unwrap()
        .copy_from_host(&[2, 4, 6, 8])
        .unwrap();
    let sequence = buffer.into_primitive_sequence();
    let values = std::thread::spawn(move || {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        get_primitive_sequence_read_handle::<u32>(&sequence, &stream)
            .unwrap()
            .to_host_vec()
            .unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(values, vec![2, 4, 6, 8]);
}

#[test]
fn input_adapter_promotes_host_storage_without_changing_the_source() {
    for default_stream in [true, false] {
        let context = CudaContext::new(0).unwrap();
        let stream = if default_stream {
            context.default_stream()
        } else {
            context.new_stream().unwrap()
        };
        for buffer in [
            rosidl_runtime_rs::Buffer::from(vec![2u8, 4, 6, 8]),
            cpu_buffer(&[2, 4, 6, 8]).into_buffer(),
        ] {
            let before = buffer.as_sequence().rosidl_buffer_ptr();
            let read = cuda_buffer_rs::from_input_buffer::<u8>(&buffer, &stream).unwrap();
            assert_eq!(read.to_host_vec().unwrap(), vec![2, 4, 6, 8]);
            assert_eq!(read.stream().cu_stream(), stream.cu_stream());
            drop(read);
            assert_eq!(buffer.backend_name().unwrap(), "cpu");
            assert_eq!(buffer.as_sequence().rosidl_buffer_ptr(), before);
            assert_eq!(buffer.to_vec().unwrap(), vec![2, 4, 6, 8]);
        }
        context.check_err().unwrap();
    }
}

#[test]
fn output_adapter_promotes_the_message_field_and_finalizes_one_write() {
    let context = CudaContext::new(0).unwrap();
    let producer = context.default_stream();
    let consumer = context.new_stream().unwrap();
    for mut buffer in [
        rosidl_runtime_rs::Buffer::from(vec![255u8; 4]),
        cpu_buffer(&[255; 4]).into_buffer(),
    ] {
        {
            let mut write =
                cuda_buffer_rs::from_output_buffer::<u8>(&mut buffer, &producer).unwrap();
            write.copy_from_host(&[3, 6, 9, 12]).unwrap();
        }
        assert_eq!(buffer.backend_name().unwrap(), "cuda");
        let owner = buffer.as_sequence().rosidl_buffer_ptr();
        let read = cuda_buffer_rs::from_input_buffer::<u8>(&buffer, &consumer).unwrap();
        assert_eq!(read.to_host_vec().unwrap(), vec![3, 6, 9, 12]);
        drop(read);
        assert!(cuda_buffer_rs::from_output_buffer::<u8>(&mut buffer, &producer).is_err());
        assert_eq!(buffer.as_sequence().rosidl_buffer_ptr(), owner);
    }
    context.check_err().unwrap();
}

#[test]
fn adapter_layout_errors_leave_host_fields_unchanged() {
    let stream = CudaContext::new(0).unwrap().default_stream();
    for values in [vec![], vec![5u8; 7]] {
        let mut buffer = rosidl_runtime_rs::Buffer::from(values.clone());
        assert!(cuda_buffer_rs::from_input_buffer::<u32>(&buffer, &stream).is_err());
        assert!(cuda_buffer_rs::from_output_buffer::<u32>(&mut buffer, &stream).is_err());
        assert_eq!(buffer.as_slice().unwrap(), values);
    }
    let mut buffer = rosidl_runtime_rs::Buffer::from(vec![5u8; 4]);
    assert!(cuda_buffer_rs::from_input_buffer::<()>(&buffer, &stream).is_err());
    assert!(cuda_buffer_rs::from_output_buffer::<()>(&mut buffer, &stream).is_err());
    assert_eq!(buffer.as_slice().unwrap(), &[5; 4]);
}

#[test]
fn promoted_input_releases_stream_and_context_after_early_drop() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let baseline = std::sync::Arc::strong_count(&stream);
    let mut buffer = rosidl_runtime_rs::Buffer::from(vec![7u8; 64]);
    for _ in 0..32 {
        let read = cuda_buffer_rs::from_input_buffer::<u8>(&buffer, &stream).unwrap();
        assert_eq!(std::sync::Arc::strong_count(&stream), baseline + 1);
        drop(read);
        buffer.as_mut_slice().unwrap().fill(9);
    }
    stream.synchronize().unwrap();
    assert_eq!(std::sync::Arc::strong_count(&stream), baseline);
    context.check_err().unwrap();
}
