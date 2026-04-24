# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DLPack-aligned tensor message.
#
# Field layout matches DLTensor from https://dmlc.github.io/dlpack/latest/
# so that any DLPack-compatible framework (PyTorch, TensorFlow, JAX, CuPy,
# ONNX Runtime, TensorRT, MXNet, RAPIDS, ...) can round-trip losslessly
# through a thin wrapper.
#
#   dtype  = {dtype_code, dtype_bits, dtype_lanes}   (DLDataType)
#   device = {device_type, device_id}                (DLDevice)
#   shape / strides / byte_offset / data             (DLTensor)

# DLDataType. dtype_code values follow DLPack's DLDataTypeCode:
#   0 = Int, 1 = UInt, 2 = Float, 3 = OpaqueHandle, 4 = BFloat,
#   5 = Complex, 6 = Bool.
# `dtype_bits` is the scalar bit width (8, 16, 32, 64, ...).
# `dtype_lanes` is the vector lane count (1 for a plain scalar).
uint8  dtype_code
uint8  dtype_bits
uint16 dtype_lanes

# DLDevice. device_type values follow DLPack's DLDeviceType:
#   1 = CPU, 2 = CUDA, 3 = CUDAHost (pinned), 10 = ROCm, 13 = CUDAManaged,
#   14 = OneAPI, ...
# `device_id` selects the device ordinal (e.g. GPU index).
int32 device_type
int32 device_id

# DLTensor shape / stride / offset.
# `strides` may be empty, which (per DLPack convention) means the tensor is
# contiguous and the receiver should infer row-major strides from `shape`.
# `byte_offset` is the offset into `data` where the tensor's first element
# lives; nonzero values allow zero-copy transport of views / sliced tensors
# that share underlying storage.
int64[]  shape
int64[]  strides
uint64   byte_offset

# Underlying storage. May be larger than `prod(shape) * dtype_bits * dtype_lanes / 8`
# if the message carries a view into a larger allocation.
# Whichever buffer backend is registered (cuda_buffer_backend for GPU
# zero-copy, CPU fallback otherwise) owns the bytes.
uint8[] data
