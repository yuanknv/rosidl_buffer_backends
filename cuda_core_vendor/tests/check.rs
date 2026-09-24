// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = cuda_core::CudaContext::new(0)?;
    context.new_stream()?.synchronize()?;
    Ok(())
}
