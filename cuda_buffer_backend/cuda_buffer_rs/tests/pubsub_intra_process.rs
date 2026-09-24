// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

mod common;

#[test]
fn cuda_image_intra_process() {
    common::run("cuda_image_intra_process", false, 3, false);
}
