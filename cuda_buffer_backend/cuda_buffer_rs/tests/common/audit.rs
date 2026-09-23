// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

static COPIED_BYTES: AtomicUsize = AtomicUsize::new(0);
static IMPORTS: Mutex<Vec<u64>> = Mutex::new(Vec::new());
static MAPPINGS: Mutex<Vec<(u64, usize, bool)>> = Mutex::new(Vec::new());

pub fn copied_bytes() -> usize {
    COPIED_BYTES.load(Ordering::SeqCst)
}

pub fn is_imported(pointer: *const u8, size: usize) -> bool {
    let address = pointer as u64;
    MAPPINGS
        .lock()
        .unwrap()
        .iter()
        .any(|&(base, len, imported)| {
            imported && address >= base && address + size as u64 <= base + len as u64
        })
}

#[link(name = "dl")]
extern "C" {
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

// The test executable exports these symbols ahead of the native CUDA libraries.
macro_rules! native_call {
    ($name:literal, $signature:ty, $($arg:expr),*) => {{
        // SAFETY: RTLD_NEXT resolves the CUDA function with the declared C ABI.
        let address = unsafe { dlsym((-1isize) as *mut c_void, concat!($name, "\0").as_ptr().cast()) };
        assert!(!address.is_null(), "missing CUDA symbol {}", $name);
        let function: $signature = unsafe { std::mem::transmute(address) };
        unsafe { function($($arg),*) }
    }};
}

#[no_mangle]
unsafe extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: i32,
    stream: *mut c_void,
) -> i32 {
    COPIED_BYTES.fetch_add(bytes, Ordering::SeqCst);
    native_call!(
        "cudaMemcpyAsync",
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, *mut c_void) -> i32,
        dst,
        src,
        bytes,
        kind,
        stream
    )
}

#[no_mangle]
unsafe extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: i32,
) -> i32 {
    COPIED_BYTES.fetch_add(bytes, Ordering::SeqCst);
    native_call!(
        "cudaMemcpy",
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
        dst,
        src,
        bytes,
        kind
    )
}

#[no_mangle]
unsafe extern "C" fn cuMemImportFromShareableHandle(
    handle: *mut u64,
    os_handle: *mut c_void,
    kind: i32,
) -> i32 {
    let status = native_call!(
        "cuMemImportFromShareableHandle",
        unsafe extern "C" fn(*mut u64, *mut c_void, i32) -> i32,
        handle,
        os_handle,
        kind
    );
    if status == 0 {
        IMPORTS.lock().unwrap().push(unsafe { *handle });
    }
    status
}

#[no_mangle]
unsafe extern "C" fn cuMemMap(
    address: u64,
    bytes: usize,
    offset: usize,
    handle: u64,
    flags: u64,
) -> i32 {
    let status = native_call!(
        "cuMemMap",
        unsafe extern "C" fn(u64, usize, usize, u64, u64) -> i32,
        address,
        bytes,
        offset,
        handle,
        flags
    );
    if status == 0 {
        let imported = IMPORTS.lock().unwrap().contains(&handle);
        MAPPINGS.lock().unwrap().push((address, bytes, imported));
    }
    status
}

#[no_mangle]
unsafe extern "C" fn cuMemUnmap(address: u64, bytes: usize) -> i32 {
    let status = native_call!(
        "cuMemUnmap",
        unsafe extern "C" fn(u64, usize) -> i32,
        address,
        bytes
    );
    if status == 0 {
        MAPPINGS
            .lock()
            .unwrap()
            .retain(|&(base, _, _)| base != address);
    }
    status
}

#[no_mangle]
unsafe extern "C" fn cuMemRelease(handle: u64) -> i32 {
    let status = native_call!("cuMemRelease", unsafe extern "C" fn(u64) -> i32, handle);
    if status == 0 {
        IMPORTS
            .lock()
            .unwrap()
            .retain(|&imported| imported != handle);
    }
    status
}
