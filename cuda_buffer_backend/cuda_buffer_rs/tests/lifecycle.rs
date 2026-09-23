// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use cuda_buffer_rs::{CudaBuffer, CudaStream, ErrorKind};

#[test]
fn raw_guards_share_device_memory_and_finalize_one_write() {
    let internal = cuda_buffer_rs::internal_stream().unwrap();
    assert!(!internal.as_raw().is_null());
    assert!(!internal.is_internal());
    assert!(CudaStream::INTERNAL.is_internal());
    for stream in [internal, CudaStream::INTERNAL] {
        let mut buffer = CudaBuffer::allocate(2048).unwrap();
        assert_eq!(buffer.len(), 2048);
        assert!(!buffer.is_empty());
        assert!(buffer.is_cuda_backed());
        let written = {
            let guard = buffer.write(stream).unwrap();
            assert_eq!(guard.len(), 2048);
            assert!(!guard.device_ptr().is_null());
            guard.device_ptr()
        };
        let error = buffer.write(stream).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Cuda);
        assert!(error.message.contains("finalized"));
        let guard = buffer.read(stream).unwrap();
        assert_eq!(guard.len(), 2048);
        assert_eq!(guard.device_ptr(), written.cast_const());
    }
}

#[test]
fn raw_and_sequence_conversions_transfer_ownership() {
    assert!(unsafe { CudaBuffer::from_raw(std::ptr::null_mut(), 0) }.is_none());
    let raw = CudaBuffer::allocate(512).unwrap().into_raw();
    assert!(!raw.is_null());
    // SAFETY: into_raw transferred the sole native owner with this byte length.
    let buffer = unsafe { CudaBuffer::from_raw(raw, 512) }.unwrap();
    assert_eq!(buffer.len(), 512);
    assert!(buffer.is_cuda_backed());
    let sequence = buffer.into_primitive_sequence();
    assert_eq!(sequence.rosidl_buffer_ptr(), Some(raw));
    assert_eq!(sequence.len(), 512);
    let guard = cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).unwrap();
    assert_eq!(guard.len(), 512);
    assert!(!guard.device_ptr().is_null());
    drop(guard);
    let reclaimed = CudaBuffer::from_primitive_sequence(sequence).unwrap();
    assert_eq!(reclaimed.len(), 512);
    assert!(reclaimed.is_cuda_backed());
    assert_eq!(
        reclaimed.into_buffer().as_sequence().rosidl_buffer_ptr(),
        Some(raw)
    );
}

#[test]
fn empty_buffers_reject_guards_and_errors_survive_later_ffi_calls() {
    let mut buffer = CudaBuffer::allocate(0).unwrap();
    assert!(buffer.is_empty());
    let error = buffer.read(CudaStream::INTERNAL).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidArgument);
    let message = error.message.clone();
    assert!(!message.is_empty());
    assert!(std::error::Error::source(&error).is_none());
    assert_eq!(
        buffer.write(CudaStream::INTERNAL).unwrap_err().kind,
        ErrorKind::InvalidArgument
    );
    let _stream = cuda_buffer_rs::internal_stream().unwrap();
    assert_eq!(error.message, message);
    assert!(error.to_string().contains(&message));
    assert!(!unsafe { cuda_buffer_rs::is_cuda_backed(std::ptr::null()) });
}

#[test]
fn normal_sequences_are_returned_on_rejected_adoption() {
    let sequence = rosidl_runtime_rs::PrimitiveSequence::from(&[1u8, 2, 3][..]);
    let error =
        cuda_buffer_rs::read_primitive_sequence(&sequence, CudaStream::INTERNAL).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidArgument);
    let sequence = CudaBuffer::from_primitive_sequence(sequence).unwrap_err();
    assert_eq!(sequence.as_slice(), &[1, 2, 3]);
}
