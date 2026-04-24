// Copyright 2026 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TORCH_TENSOR_BRIDGE__TORCH_TENSOR_BRIDGE_HPP_
#define TORCH_TENSOR_BRIDGE__TORCH_TENSOR_BRIDGE_HPP_

#include <c10/core/StreamGuard.h>
#include <rcutils/logging_macros.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rosidl_buffer/buffer.hpp"
#include "torch_tensor_msgs/msg/tensor.hpp"

#if __has_include("cuda_buffer/cuda_buffer_api.hpp")
#include <c10/cuda/CUDAStream.h>
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#define TORCH_TENSOR_BRIDGE_HAS_CUDA
#endif

namespace torch_tensor_bridge
{

using TensorMsg = torch_tensor_msgs::msg::Tensor;

// DLPack constants replicated locally so the wire-format meaning is explicit
// and doesn't depend on <ATen/dlpack.h> or a DLPack C header being in scope.
// Values match https://dmlc.github.io/dlpack/latest/ exactly.
namespace dlpack
{

enum DLDataTypeCode : uint8_t
{
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
  kDLOpaqueHandle = 3,
  kDLBfloat = 4,
  kDLComplex = 5,
  kDLBool = 6,
};

enum DLDeviceType : int32_t
{
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLROCMHost = 11,
  kDLExtDev = 12,
  kDLCUDAManaged = 13,
  kDLOneAPI = 14,
  kDLWebGPU = 15,
  kDLHexagon = 16,
};

}  // namespace dlpack

/// DLPack-equivalent scalar dtype triple.
struct DLDataType
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

/// DLPack-equivalent device tuple.
struct DLDevice
{
  int32_t type;
  int32_t id;
};

// ---------------------------------------------------------------------------
// dtype conversions (at::ScalarType <-> DLDataType)
// ---------------------------------------------------------------------------

inline DLDataType dl_dtype_from_scalar(at::ScalarType t)
{
  switch (t) {
    case at::kByte:     return {dlpack::kDLUInt, 8, 1};
    case at::kChar:     return {dlpack::kDLInt, 8, 1};
    case at::kShort:    return {dlpack::kDLInt, 16, 1};
    case at::kInt:      return {dlpack::kDLInt, 32, 1};
    case at::kLong:     return {dlpack::kDLInt, 64, 1};
    case at::kHalf:     return {dlpack::kDLFloat, 16, 1};
    case at::kBFloat16: return {dlpack::kDLBfloat, 16, 1};
    case at::kFloat:    return {dlpack::kDLFloat, 32, 1};
    case at::kDouble:   return {dlpack::kDLFloat, 64, 1};
    case at::kBool:     return {dlpack::kDLBool, 8, 1};
    default:
      throw std::runtime_error(
              "torch_tensor_bridge: unsupported at::ScalarType for DLPack encoding");
  }
}

inline at::ScalarType scalar_from_dl_dtype(DLDataType d)
{
  if (d.lanes != 1) {
    throw std::runtime_error(
            "torch_tensor_bridge: dtype_lanes != 1 not representable as at::ScalarType");
  }
  switch (d.code) {
    case dlpack::kDLUInt:
      if (d.bits == 8) {return at::kByte;}
      break;
    case dlpack::kDLInt:
      switch (d.bits) {
        case 8: return at::kChar;
        case 16: return at::kShort;
        case 32: return at::kInt;
        case 64: return at::kLong;
      }
      break;
    case dlpack::kDLFloat:
      switch (d.bits) {
        case 16: return at::kHalf;
        case 32: return at::kFloat;
        case 64: return at::kDouble;
      }
      break;
    case dlpack::kDLBfloat:
      if (d.bits == 16) {return at::kBFloat16;}
      break;
    case dlpack::kDLBool:
      if (d.bits == 8) {return at::kBool;}
      break;
  }
  throw std::runtime_error(
          "torch_tensor_bridge: unsupported DLDataType (code=" +
          std::to_string(static_cast<int>(d.code)) +
          ", bits=" + std::to_string(static_cast<int>(d.bits)) +
          ", lanes=" + std::to_string(d.lanes) + ")");
}

/// Size, in bytes, of a single DLPack element (scalar * lanes).
inline size_t dl_dtype_bytesize(DLDataType d)
{
  return (static_cast<size_t>(d.bits) * d.lanes + 7) / 8;
}

// ---------------------------------------------------------------------------
// Message field accessors (optional convenience)
// ---------------------------------------------------------------------------

inline DLDataType get_dtype(const TensorMsg & m)
{
  return {m.dtype_code, m.dtype_bits, m.dtype_lanes};
}

inline void set_dtype(TensorMsg & m, DLDataType d)
{
  m.dtype_code = d.code;
  m.dtype_bits = d.bits;
  m.dtype_lanes = d.lanes;
}

inline DLDevice get_device(const TensorMsg & m)
{
  return {m.device_type, m.device_id};
}

inline void set_device(TensorMsg & m, DLDevice d)
{
  m.device_type = d.type;
  m.device_id = d.id;
}

// ---------------------------------------------------------------------------
// Stream helpers
// ---------------------------------------------------------------------------

namespace detail
{

inline std::optional<c10::Stream> get_non_default_stream()
{
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  if (torch::cuda::is_available()) {
    return c10::cuda::getStreamFromPool();
  }
#endif
  return std::nullopt;
}

inline c10::DeviceType default_device()
{
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  if (torch::cuda::is_available()) {
    return c10::kCUDA;
  }
#endif
  return c10::kCPU;
}

inline std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> strides(shape.size());
  int64_t s = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = s;
    s *= shape[i];
  }
  return strides;
}

inline int64_t numel_of(const std::vector<int64_t> & shape)
{
  int64_t n = 1;
  for (auto d : shape) {
    if (d < 0) {
      throw std::runtime_error("torch_tensor_bridge: negative shape dimension");
    }
    n *= d;
  }
  return n;
}

}  // namespace detail

/// RAII guard that sets a non-default CUDA stream for the current scope.
class StreamGuard
{
public:
  StreamGuard()
  : guard_(detail::get_non_default_stream()) {}

private:
  c10::OptionalStreamGuard guard_;
};

inline StreamGuard set_stream() {return StreamGuard();}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

/// Allocate a Tensor message with DLPack metadata populated and a pre-sized
/// `data` buffer (CUDA-backed when available, else CPU). The storage is sized
/// exactly to hold `prod(shape) * dtype.bytesize` bytes and `byte_offset` is
/// left at zero; callers needing a view into larger storage can post-size
/// `msg.data` and set `byte_offset` manually.
inline TensorMsg allocate_tensor(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::DeviceType> device = std::nullopt)
{
  c10::DeviceType dev = device.value_or(detail::default_device());
  DLDataType dl = dl_dtype_from_scalar(dtype);
  int64_t numel = detail::numel_of(shape);
  size_t byte_count = static_cast<size_t>(numel) * dl_dtype_bytesize(dl);

  TensorMsg msg;
  set_dtype(msg, dl);
  msg.shape.assign(shape.begin(), shape.end());
  auto strides = detail::contiguous_strides(shape);
  msg.strides.assign(strides.begin(), strides.end());
  msg.byte_offset = 0;

#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  if (dev == c10::kCUDA) {
    int cur = 0;
    cudaGetDevice(&cur);
    set_device(msg, {dlpack::kDLCUDA, cur});
    auto cuda_impl = std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(byte_count);
    msg.data = rosidl::Buffer<uint8_t>(std::move(cuda_impl));
    return msg;
  }
#endif
  if (dev == c10::kCPU) {
    set_device(msg, {dlpack::kDLCPU, 0});
    msg.data.resize(byte_count);
    return msg;
  }
  throw std::runtime_error(
          "torch_tensor_bridge: unsupported device type " +
          std::to_string(static_cast<int>(dev)));
}

// ---------------------------------------------------------------------------
// View construction
// ---------------------------------------------------------------------------

namespace detail
{

inline at::Tensor cpu_wrap(
  void * ptr,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype)
{
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  return strides.empty() ?
         torch::from_blob(ptr, shape, opts) :
         torch::from_blob(ptr, shape, strides, opts);
}

#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA

inline cudaStream_t current_cuda_stream(const char * ctx)
{
  cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
  if (s == nullptr) {
    RCUTILS_LOG_WARN_NAMED(
      "torch_tensor_bridge",
      "%s: current CUDA stream is the default stream. "
      "Set a non-default stream (e.g. via torch_tensor_bridge::set_stream()) "
      "for event-based synchronization.",
      ctx);
  }
  return s;
}

inline at::Tensor cuda_wrap_writable(
  rosidl::Buffer<uint8_t> & data,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype,
  uint64_t byte_offset)
{
  auto * cuda_impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(data.get_impl()));
  if (!cuda_impl) {
    throw std::runtime_error(
            "torch_tensor_bridge: from_tensor_msg (write) expected CUDA-backed data buffer");
  }
  cudaStream_t stream = current_cuda_stream("from_tensor_msg (write)");
  cuda_impl->set_stream(stream);
  auto wh = std::make_shared<cuda_buffer_backend::WriteHandle>(
    cuda_impl->get_cuda_buffer().get_write_handle(stream));
  auto * base = static_cast<uint8_t *>(wh->get_ptr());
  void * ptr = base + byte_offset;
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
  return strides.empty() ?
         torch::from_blob(ptr, shape, [wh](void *) {}, opts) :
         torch::from_blob(ptr, shape, strides, [wh](void *) {}, opts);
}

inline at::Tensor cuda_wrap_readable(
  const rosidl::Buffer<uint8_t> & data,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype,
  uint64_t byte_offset,
  bool clone)
{
  const auto * cuda_impl = dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    data.get_impl());
  if (!cuda_impl) {
    throw std::runtime_error(
            "torch_tensor_bridge: from_tensor_msg (read) expected CUDA-backed data buffer");
  }
  cudaStream_t stream = current_cuda_stream("from_tensor_msg (read)");
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
  if (clone) {
    auto rh = cuda_impl->get_cuda_buffer().get_read_handle(stream);
    auto * base = const_cast<uint8_t *>(static_cast<const uint8_t *>(rh.get_ptr()));
    void * ptr = base + byte_offset;
    at::Tensor view = strides.empty() ?
      torch::from_blob(ptr, shape, opts) :
      torch::from_blob(ptr, shape, strides, opts);
    return view.clone();
  }
  auto rh = std::make_shared<cuda_buffer_backend::ReadHandle>(
    cuda_impl->get_cuda_buffer().get_read_handle(stream));
  auto * base = const_cast<uint8_t *>(static_cast<const uint8_t *>(rh->get_ptr()));
  void * ptr = base + byte_offset;
  return strides.empty() ?
         torch::from_blob(ptr, shape, [rh](void *) {}, opts) :
         torch::from_blob(ptr, shape, strides, [rh](void *) {}, opts);
}

#endif  // TORCH_TENSOR_BRIDGE_HAS_CUDA

}  // namespace detail

/// Get a writable at::Tensor view over msg.data + msg.byte_offset.
/// The view shares memory with msg.data; the caller must ensure msg outlives
/// the returned tensor.
inline at::Tensor from_tensor_msg(TensorMsg & msg)
{
  if (msg.data.empty()) {return {};}
  std::vector<int64_t> shape(msg.shape.begin(), msg.shape.end());
  std::vector<int64_t> strides(msg.strides.begin(), msg.strides.end());
  at::ScalarType dtype = scalar_from_dl_dtype(get_dtype(msg));
  const std::string & backend = msg.data.get_backend_type();
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  if (backend == "cuda") {
    return detail::cuda_wrap_writable(
      msg.data, shape, strides, dtype, msg.byte_offset);
  }
#endif
  if (backend == "cpu") {
    auto * base = static_cast<uint8_t *>(msg.data.data());
    void * ptr = base + msg.byte_offset;
    return detail::cpu_wrap(ptr, shape, strides, dtype);
  }
  throw std::runtime_error(
          "torch_tensor_bridge: unsupported data backend '" + backend + "'");
}

/// Get a read-only at::Tensor from msg.data + msg.byte_offset.
/// \param clone If true (default), returns an independent copy. If false,
/// returns a zero-copy view (for CUDA, a ReadHandle is kept alive via the
/// tensor's deleter so event-based synchronization stays correct).
inline at::Tensor from_tensor_msg(const TensorMsg & msg, bool clone = true)
{
  if (msg.data.empty()) {return {};}
  std::vector<int64_t> shape(msg.shape.begin(), msg.shape.end());
  std::vector<int64_t> strides(msg.strides.begin(), msg.strides.end());
  at::ScalarType dtype = scalar_from_dl_dtype(get_dtype(msg));
  const std::string & backend = msg.data.get_backend_type();
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  if (backend == "cuda") {
    return detail::cuda_wrap_readable(
      msg.data, shape, strides, dtype, msg.byte_offset, clone);
  }
#endif
  if (backend == "cpu") {
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    auto * base = const_cast<uint8_t *>(static_cast<const uint8_t *>(msg.data.data()));
    void * ptr = base + msg.byte_offset;
    at::Tensor view = strides.empty() ?
      torch::from_blob(ptr, shape, opts) :
      torch::from_blob(ptr, shape, strides, opts);
    return clone ? view.clone() : view;
  }
  throw std::runtime_error(
          "torch_tensor_bridge: unsupported data backend '" + backend + "'");
}

/// Copy `tensor` into msg.data (pre-allocated, with the same device as the
/// source tensor's) and refresh msg metadata to the contiguous form.
/// `byte_offset` is reset to 0; shape/strides/dtype are overwritten.
/// Device fields are left untouched (the storage device is fixed at
/// allocation time).
inline void to_tensor_msg(TensorMsg & msg, const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }
  at::Tensor contig = tensor.contiguous();
  size_t byte_count = contig.numel() * contig.element_size();

  if (byte_count > msg.data.size()) {
    throw std::runtime_error(
            "torch_tensor_bridge::to_tensor_msg: tensor size (" +
            std::to_string(byte_count) + " bytes) exceeds allocated buffer (" +
            std::to_string(msg.data.size()) + " bytes)");
  }

  const std::string & backend = msg.data.get_backend_type();
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  if (backend == "cuda") {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto wh = cuda_buffer_backend::from_buffer(msg.data, stream);
    cudaMemcpyKind kind = contig.is_cuda() ?
      cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cuda_buffer_backend::to_buffer(
      contig.data_ptr(), byte_count, wh, stream, kind);
  } else  // NOLINT(readability/braces)
#endif
  if (backend == "cpu") {
    at::Tensor cpu_contig = contig.to(torch::kCPU).contiguous();
    std::memcpy(msg.data.data(), cpu_contig.data_ptr(), byte_count);
  } else {
    throw std::runtime_error(
            "torch_tensor_bridge::to_tensor_msg: unsupported backend '" + backend + "'");
  }

  auto sizes = contig.sizes().vec();
  auto strides = contig.strides().vec();
  msg.shape.assign(sizes.begin(), sizes.end());
  msg.strides.assign(strides.begin(), strides.end());
  set_dtype(msg, dl_dtype_from_scalar(contig.scalar_type()));
  msg.byte_offset = 0;
}

}  // namespace torch_tensor_bridge

#endif  // TORCH_TENSOR_BRIDGE__TORCH_TENSOR_BRIDGE_HPP_
