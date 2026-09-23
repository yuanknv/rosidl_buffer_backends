// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Scoped Rust access to native CUDA buffers.
//!
//! `CudaBuffer` owns the native allocation. Guards borrow it while native events
//! order GPU access. The `cuda-core` feature adds typed
//! access using retained cuda-core streams. Buffers and handles are thread-local.

#![warn(unsafe_op_in_unsafe_fn)]
#![cfg_attr(feature = "cuda-core", doc = include_str!("README.md"))]

pub mod ffi;

#[cfg(feature = "cuda-core")]
mod typed;
#[cfg(feature = "cuda-core")]
pub use typed::{
    from_input_buffer, from_output_buffer, get_primitive_sequence_read_handle, to_buffer, CopyKind,
    CudaReadHandle, CudaWriteHandle,
};

use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr::{self, NonNull};

use rosidl_runtime_rs::PrimitiveSequence;

/// A `cudaStream_t` to order buffer access on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStream(*mut c_void);

impl CudaStream {
    /// Defer to the backend's process-wide internal stream.
    pub const INTERNAL: Self = Self(ptr::null_mut());

    /// Adopt an existing `cudaStream_t`.
    ///
    /// # Safety
    ///
    /// `raw` must be null or a live `cudaStream_t` that outlives every guard
    /// acquired with it.
    pub const unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self(raw)
    }

    pub fn as_raw(self) -> *mut c_void {
        self.0
    }

    pub fn is_internal(self) -> bool {
        self.0.is_null()
    }
}

/// Classification of a `cuda_buffer` C ABI failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    BadAlloc,
    Cuda,
    Other,
}

/// Error returned by the CUDA buffer bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBufferError {
    pub kind: ErrorKind,
    pub message: String,
}

impl fmt::Display for CudaBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for CudaBufferError {}

/// Result of a CUDA buffer operation.
pub type Result<T> = std::result::Result<T, CudaBufferError>;

fn check(ret: ffi::cuda_buffer_ret_t) -> Result<()> {
    let kind = match ret {
        ffi::CUDA_BUFFER_RET_OK => return Ok(()),
        ffi::CUDA_BUFFER_RET_INVALID_ARGUMENT => ErrorKind::InvalidArgument,
        ffi::CUDA_BUFFER_RET_BAD_ALLOC => ErrorKind::BadAlloc,
        ffi::CUDA_BUFFER_RET_CUDA_ERROR => ErrorKind::Cuda,
        _ => ErrorKind::Other,
    };
    Err(CudaBufferError {
        kind,
        message: last_error_message(),
    })
}

fn last_error_message() -> String {
    // SAFETY: the ABI returns a non-null, thread-local C string.
    unsafe { CStr::from_ptr(ffi::cuda_buffer_error_message()) }
        .to_string_lossy()
        .into_owned()
}

fn corrupt_abi(message: &str) -> CudaBufferError {
    CudaBufferError {
        kind: ErrorKind::Other,
        message: message.to_string(),
    }
}

/// Get the backend's process-wide internal CUDA stream.
pub fn internal_stream() -> Result<CudaStream> {
    let mut raw = ptr::null_mut();
    check(unsafe { ffi::cuda_buffer_internal_stream(&mut raw) })?;
    Ok(unsafe { CudaStream::from_raw(raw) })
}

/// Report whether an opaque `rosidl::Buffer<uint8_t> *` uses the CUDA backend.
///
/// # Safety
///
/// `buffer` must be null or point to a live `rosidl::Buffer<uint8_t>`.
pub unsafe fn is_cuda_backed(buffer: *const c_void) -> bool {
    unsafe { ffi::cuda_buffer_is_cuda_backed(buffer) }
}

/// Allocate uninitialized CUDA bytes for a generated message field.
///
/// Uses the native CUDA pool without acquiring a handle or copying data.
/// Initialize the output through `from_output_buffer` before publishing it.
pub fn allocate_buffer(byte_count: usize) -> Result<rosidl_runtime_rs::Buffer<u8>> {
    CudaBuffer::allocate(byte_count).map(CudaBuffer::into_buffer)
}

/// Owning handle to scoped read access, released on drop.
///
/// Prefer [`CudaBuffer::read`], which ties the handle to a borrow of the buffer.
#[must_use]
pub struct ReadHandle<'a> {
    raw: NonNull<ffi::cuda_buffer_read_handle_t>,
    _owner: PhantomData<&'a ()>,
}

impl ReadHandle<'_> {
    /// Acquire read access to an opaque `rosidl::Buffer<uint8_t> *`.
    ///
    /// A non-CUDA buffer is promoted with a host-to-device copy; the promotion is
    /// retained by the handle and never becomes the caller's to release.
    ///
    /// # Safety
    ///
    /// `buffer` must point to a live `rosidl::Buffer<uint8_t>` that outlives the
    /// returned handle. Enqueue work on `stream` before dropping the handle;
    /// the stream must remain live until then.
    pub unsafe fn acquire(buffer: *const c_void, stream: CudaStream) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { ffi::cuda_buffer_acquire_read(buffer, stream.as_raw(), &mut raw) })?;
        NonNull::new(raw)
            .map(|raw| Self {
                raw,
                _owner: PhantomData,
            })
            .ok_or_else(|| corrupt_abi("read acquisition returned a null handle"))
    }

    /// Device pointer to the readable bytes.
    pub fn device_ptr(&self) -> *const u8 {
        unsafe { ffi::cuda_buffer_read_handle_data(self.raw.as_ptr()) }
    }

    /// Number of readable bytes.
    pub fn len(&self) -> usize {
        unsafe { ffi::cuda_buffer_read_handle_size(self.raw.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for ReadHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadHandle")
            .field("device_ptr", &self.device_ptr())
            .field("len", &self.len())
            .finish()
    }
}

impl Drop for ReadHandle<'_> {
    fn drop(&mut self) {
        unsafe { ffi::cuda_buffer_read_handle_destroy(self.raw.as_ptr()) };
    }
}

/// Owning handle to scoped write access, released on drop.
///
/// Prefer [`CudaBuffer::write`], which ties the handle to a mutable borrow of the
/// buffer.
#[must_use]
pub struct WriteHandle {
    raw: NonNull<ffi::cuda_buffer_write_handle_t>,
}

impl WriteHandle {
    /// Acquire write access to the opaque `rosidl::Buffer<uint8_t> *` in `buffer`.
    ///
    /// When the buffer is not CUDA-backed it is promoted, and on success
    /// `*buffer` is replaced with a newly allocated CUDA-backed buffer that the
    /// caller now owns in addition to the pointer it passed in. The promoted
    /// contents are uninitialized. On failure `*buffer` is unchanged.
    ///
    /// # Safety
    ///
    /// `*buffer` must point to a live `rosidl::Buffer<uint8_t>` through all uses
    /// of the returned handle. Submit work before the buffer is published, read,
    /// or destroyed. The stream must remain live through handle cleanup.
    pub unsafe fn acquire(buffer: &mut *mut c_void, stream: CudaStream) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { ffi::cuda_buffer_acquire_write(buffer, stream.as_raw(), &mut raw) })?;
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(|| corrupt_abi("write acquisition returned a null handle"))
    }

    /// Device pointer to the writable bytes.
    pub fn device_ptr(&self) -> *mut u8 {
        unsafe { ffi::cuda_buffer_write_handle_data(self.raw.as_ptr()) }
    }

    /// Number of writable bytes.
    pub fn len(&self) -> usize {
        unsafe { ffi::cuda_buffer_write_handle_size(self.raw.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Debug for WriteHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteHandle")
            .field("device_ptr", &self.device_ptr())
            .field("len", &self.len())
            .finish()
    }
}

impl Drop for WriteHandle {
    fn drop(&mut self) {
        unsafe { ffi::cuda_buffer_write_handle_destroy(self.raw.as_ptr()) };
    }
}

/// An owned `rosidl::Buffer<uint8_t>` allocated or adopted from the C++ backend.
///
/// Drop releases the owner through `rosidl_buffer_uint8_destroy`.
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<cuda_buffer_rs::CudaBuffer>();
/// ```
#[must_use]
pub struct CudaBuffer {
    raw: NonNull<c_void>,
    len: usize,
}

impl CudaBuffer {
    /// Transfers the native owner into a backend-neutral message buffer.
    pub fn into_buffer(self) -> rosidl_runtime_rs::Buffer<u8> {
        self.into_primitive_sequence().into()
    }

    /// Allocate `len` CUDA-backed bytes. The contents are uninitialized.
    pub fn allocate(len: usize) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { ffi::cuda_buffer_allocate(len, &mut raw) })?;
        NonNull::new(raw)
            .map(|raw| Self { raw, len })
            .ok_or_else(|| corrupt_abi("allocation returned a null buffer"))
    }

    /// Take ownership of an existing opaque `rosidl::Buffer<uint8_t> *`.
    ///
    /// # Safety
    ///
    /// `raw` must be a live buffer that was heap-allocated by the ROS IDL buffer
    /// stack, `len` must be its byte length, and no other owner may exist.
    pub unsafe fn from_raw(raw: *mut c_void, len: usize) -> Option<Self> {
        NonNull::new(raw).map(|raw| Self { raw, len })
    }

    /// Borrow the opaque buffer pointer without releasing ownership.
    pub fn as_ptr(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    /// Release ownership, returning the opaque buffer pointer.
    ///
    /// The caller becomes responsible for destroying it, either through
    /// [`CudaBuffer::from_raw`] or the C ABI.
    pub fn into_raw(self) -> *mut c_void {
        let raw = self.raw.as_ptr();
        std::mem::forget(self);
        raw
    }

    /// Transfers this allocation into an RMW-native `uint8[]` field.
    pub fn into_primitive_sequence(self) -> PrimitiveSequence<u8> {
        let len = self.len;
        let raw = self.into_raw();
        unsafe { PrimitiveSequence::from_owned_rosidl_buffer(raw, len) }
            .expect("CudaBuffer always contains a non-null Buffer pointer")
    }

    /// Takes a CUDA-backed allocation out of an RMW-native `uint8[]` field.
    pub fn from_primitive_sequence(
        sequence: PrimitiveSequence<u8>,
    ) -> std::result::Result<Self, PrimitiveSequence<u8>> {
        let Some(raw) = sequence.rosidl_buffer_ptr() else {
            return Err(sequence);
        };
        if !unsafe { is_cuda_backed(raw) } {
            return Err(sequence);
        }
        let len = sequence.len();
        let raw = sequence.into_owned_rosidl_buffer()?;
        Ok(unsafe { Self::from_raw(raw, len) }.expect("Buffer pointer was verified as non-null"))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Report whether the buffer currently uses the CUDA backend.
    pub fn is_cuda_backed(&self) -> bool {
        unsafe { ffi::cuda_buffer_is_cuda_backed(self.raw.as_ptr()) }
    }

    /// Acquire scoped read access ordered on `stream`.
    ///
    /// The owner remains borrowed through automatic read-handle cleanup:
    /// ```compile_fail
    /// use cuda_buffer_rs::{CudaBuffer, CudaStream};
    /// let buffer = CudaBuffer::allocate(16).unwrap();
    /// let read = buffer.read(CudaStream::INTERNAL).unwrap();
    /// let _ = read.device_ptr();
    /// drop(buffer);
    /// ```
    pub fn read(&self, stream: CudaStream) -> Result<CudaReadGuard<'_>> {
        let handle = unsafe { ReadHandle::acquire(self.raw.as_ptr(), stream) }?;
        Ok(handle)
    }

    /// Acquire scoped write access ordered on `stream`.
    ///
    /// ```compile_fail
    /// use cuda_buffer_rs::{CudaBuffer, CudaStream};
    /// let mut buffer = CudaBuffer::allocate(16).unwrap();
    /// let write = buffer.write(CudaStream::INTERNAL).unwrap();
    /// let read = buffer.read(CudaStream::INTERNAL).unwrap();
    /// drop(write);
    /// drop(read);
    /// ```
    ///
    /// If the buffer was adopted from non-CUDA storage it is promoted and this
    /// `CudaBuffer` takes ownership of the promoted allocation, destroying the
    /// previous one. [`CudaBuffer::as_ptr`] then reports the new pointer, which
    /// is the one that must be published.
    pub fn write(&mut self, stream: CudaStream) -> Result<CudaWriteGuard<'_>> {
        let mut slot = self.raw.as_ptr();
        let handle = unsafe { WriteHandle::acquire(&mut slot, stream) }?;
        if slot != self.raw.as_ptr() {
            let promoted = NonNull::new(slot)
                .ok_or_else(|| corrupt_abi("write acquisition returned a null promoted buffer"))?;
            let previous = std::mem::replace(&mut self.raw, promoted);
            unsafe { ffi::rosidl_buffer_uint8_destroy(previous.as_ptr()) };
        }
        Ok(CudaWriteGuard {
            handle,
            _owner: PhantomData,
        })
    }
}

impl fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("ptr", &self.raw.as_ptr())
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe { ffi::rosidl_buffer_uint8_destroy(self.raw.as_ptr()) };
    }
}

/// Scoped read access tied to a borrow of its [`CudaBuffer`].
pub type CudaReadGuard<'a> = ReadHandle<'a>;

/// Scoped write access tied to a mutable borrow of its [`CudaBuffer`].
#[derive(Debug)]
pub struct CudaWriteGuard<'a> {
    handle: WriteHandle,
    _owner: PhantomData<&'a mut ()>,
}

impl CudaWriteGuard<'_> {
    pub fn device_ptr(&self) -> *mut u8 {
        self.handle.device_ptr()
    }

    pub fn len(&self) -> usize {
        self.handle.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handle.is_empty()
    }
}

/// Acquires CUDA read access directly from an RMW-native `uint8[]` field.
pub fn read_primitive_sequence(
    sequence: &PrimitiveSequence<u8>,
    stream: CudaStream,
) -> Result<CudaReadGuard<'_>> {
    let raw = sequence
        .rosidl_buffer_ptr()
        .ok_or_else(|| CudaBufferError {
            kind: ErrorKind::InvalidArgument,
            message: "primitive sequence is not Buffer-backed".into(),
        })?;
    let handle = unsafe { ReadHandle::acquire(raw, stream) }?;
    Ok(handle)
}

#[cfg(test)]
mod tests;
