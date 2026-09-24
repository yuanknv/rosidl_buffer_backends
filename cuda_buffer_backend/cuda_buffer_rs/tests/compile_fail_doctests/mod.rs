// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Buffers are thread-local.
//! ```compile_fail
//! fn require_send<T: Send>() {}
//! require_send::<cuda_buffer_rs::CudaBuffer>();
//! ```
//!
//! Read guards retain the owner borrow through cleanup.
//! ```compile_fail
//! use cuda_buffer_rs::{CudaBuffer, CudaStream};
//! let buffer = CudaBuffer::allocate(16).unwrap();
//! let read = buffer.read(CudaStream::INTERNAL).unwrap();
//! let _ = read.device_ptr();
//! drop(buffer);
//! ```
//!
//! Write guards exclude simultaneous reads.
//! ```compile_fail
//! use cuda_buffer_rs::{CudaBuffer, CudaStream};
//! let mut buffer = CudaBuffer::allocate(16).unwrap();
//! let write = buffer.write(CudaStream::INTERNAL).unwrap();
//! let read = buffer.read(CudaStream::INTERNAL).unwrap();
//! drop(write);
//! drop(read);
//! ```
//!

#[cfg(feature = "cuda-core")]
mod typed {
    //! Read handles have no mutable device access.
    //! ```compile_fail
    //! use cuda_buffer_rs::CudaReadHandle;
    //! fn mutate(handle: &mut CudaReadHandle<'_, u32>) {
    //!     handle.as_device_buffer_mut();
    //! }
    //! ```
    //!
    //! Owning CUDA allocations cannot be extracted through safe code.
    //! ```compile_fail
    //! use cuda_buffer_rs::CudaWriteHandle;
    //! use cuda_core::DeviceBuffer;
    //! fn steal(handle: &mut CudaWriteHandle<'_, u32>, replacement: DeviceBuffer<u32>) {
    //!     let stolen = std::mem::replace(handle.as_device_buffer_mut(), replacement);
    //! }
    //! ```
    //!
    //! Typed read handles borrow their owner.
    //! ```compile_fail
    //! use cuda_buffer_rs::CudaBuffer;
    //! use cuda_core::CudaContext;
    //! let stream = CudaContext::new(0).unwrap().default_stream();
    //! let buffer = CudaBuffer::allocate(16).unwrap();
    //! let handle = buffer.get_read_handle::<u32>(&stream).unwrap();
    //! drop(buffer);
    //! let _ = handle.to_host_vec();
    //! ```
    //!
    //! Element types must implement DeviceCopy.
    //! ```compile_fail
    //! use cuda_buffer_rs::CudaBuffer;
    //! use cuda_core::CudaContext;
    //! let stream = CudaContext::new(0).unwrap().default_stream();
    //! let mut buffer = CudaBuffer::allocate(16).unwrap();
    //! let _ = buffer.get_write_handle::<bool>(&stream);
    //! ```
    //!
    //! Typed write handles exclude simultaneous reads.
    //! ```compile_fail
    //! use cuda_buffer_rs::CudaBuffer;
    //! use cuda_core::CudaContext;
    //! let stream = CudaContext::new(0).unwrap().default_stream();
    //! let mut buffer = CudaBuffer::allocate(16).unwrap();
    //! let write = buffer.get_write_handle::<u32>(&stream).unwrap();
    //! let read = buffer.get_read_handle::<u32>(&stream).unwrap();
    //! drop(write);
    //! drop(read);
    //! ```
    //!
    //! Sequence read handles borrow their owner.
    //! ```compile_fail
    //! use cuda_buffer_rs::{CudaBuffer, get_primitive_sequence_read_handle};
    //! use cuda_core::CudaContext;
    //! let stream = CudaContext::new(0).unwrap().default_stream();
    //! let sequence = CudaBuffer::allocate(16).unwrap().into_primitive_sequence();
    //! let read = get_primitive_sequence_read_handle::<u32>(&sequence, &stream).unwrap();
    //! drop(sequence);
    //! drop(read);
    //! ```
    //!
    //! Raw-pointer copies require unsafe.
    //! ```compile_fail
    //! use cuda_buffer_rs::{to_buffer, CopyKind, CudaWriteHandle};
    //! use cuda_core::CudaStream;
    //! use std::sync::Arc;
    //! fn copy(source: *const std::ffi::c_void, output: &mut CudaWriteHandle<'_, u8>, stream: &Arc<CudaStream>) {
    //!     to_buffer(source, 4, output, stream, CopyKind::DeviceToDevice).unwrap();
    //! }
    //! ```
    //!
    //! Input handles retain the message-field borrow through cleanup.
    //! ```compile_fail
    //! use cuda_buffer_rs::from_input_buffer;
    //! use cuda_core::CudaContext;
    //! let stream = CudaContext::new(0).unwrap().default_stream();
    //! let data = rosidl_runtime_rs::Buffer::from(vec![1u8; 4]);
    //! let read = from_input_buffer::<u8>(&data, &stream).unwrap();
    //! let _ = read.to_host_vec().unwrap();
    //! drop(data);
    //! ```
    //!
    //! Output handles exclude simultaneous reads of the message field.
    //! ```compile_fail
    //! use cuda_buffer_rs::{from_input_buffer, from_output_buffer};
    //! use cuda_core::CudaContext;
    //! let stream = CudaContext::new(0).unwrap().default_stream();
    //! let mut data = rosidl_runtime_rs::Buffer::from(vec![0u8; 4]);
    //! let write = from_output_buffer::<u8>(&mut data, &stream).unwrap();
    //! let read = from_input_buffer::<u8>(&data, &stream).unwrap();
    //! drop(write);
    //! drop(read);
    //! ```
    //!
}
