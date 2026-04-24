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

#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
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

// Re-export the canonical DLPack types from <dlpack.h> under our namespace
// so user code can write `torch_tensor_bridge::DLDataType` without needing
// to know about the global dlpack header. The DLDataTypeCode / DLDeviceType
// enumerators (kDLInt, kDLUInt, kDLCPU, kDLCUDA, ...) are already in the
// global namespace courtesy of <dlpack.h>, so users can just write
// `kDLCUDA` or fully qualify as `::kDLCUDA`.
using DLDataType = ::DLDataType;
using DLDevice = ::DLDevice;
using DLManagedTensor = ::DLManagedTensor;

// ---------------------------------------------------------------------------
// dtype conversions (at::ScalarType <-> DLDataType)
// ---------------------------------------------------------------------------

inline DLDataType dl_dtype_from_scalar(at::ScalarType t)
{
  switch (t) {
    case at::kByte:     return DLDataType{kDLUInt, 8, 1};
    case at::kChar:     return DLDataType{kDLInt, 8, 1};
    case at::kShort:    return DLDataType{kDLInt, 16, 1};
    case at::kInt:      return DLDataType{kDLInt, 32, 1};
    case at::kLong:     return DLDataType{kDLInt, 64, 1};
    case at::kHalf:     return DLDataType{kDLFloat, 16, 1};
    case at::kBFloat16: return DLDataType{kDLBfloat, 16, 1};
    case at::kFloat:    return DLDataType{kDLFloat, 32, 1};
    case at::kDouble:   return DLDataType{kDLFloat, 64, 1};
    case at::kBool:     return DLDataType{kDLBool, 8, 1};
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
    case kDLUInt:
      if (d.bits == 8) {return at::kByte;}
      break;
    case kDLInt:
      switch (d.bits) {
        case 8: return at::kChar;
        case 16: return at::kShort;
        case 32: return at::kInt;
        case 64: return at::kLong;
      }
      break;
    case kDLFloat:
      switch (d.bits) {
        case 16: return at::kHalf;
        case 32: return at::kFloat;
        case 64: return at::kDouble;
      }
      break;
    case kDLBfloat:
      if (d.bits == 16) {return at::kBFloat16;}
      break;
    case kDLBool:
      if (d.bits == 8) {return at::kBool;}
      break;
  }
  throw std::runtime_error(
          "torch_tensor_bridge: unsupported DLDataType (code=" +
          std::to_string(static_cast<int>(d.code)) +
          ", bits=" + std::to_string(static_cast<int>(d.bits)) +
          ", lanes=" + std::to_string(d.lanes) + ")");
}

inline size_t dl_dtype_bytesize(DLDataType d)
{
  return (static_cast<size_t>(d.bits) * d.lanes + 7) / 8;
}

// ---------------------------------------------------------------------------
// Message field accessors
// ---------------------------------------------------------------------------

/// Pack the three dtype fields of `m` back into a DLDataType struct.
inline void set_dtype(TensorMsg & m, DLDataType d)
{
  m.dtype_code = d.code;
  m.dtype_bits = d.bits;
  m.dtype_lanes = d.lanes;
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
#endif

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
    msg.device_type = static_cast<int32_t>(kDLCUDA);
    msg.device_id = cur;
    auto cuda_impl = std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(byte_count);
    msg.data = rosidl::Buffer<uint8_t>(std::move(cuda_impl));
    return msg;
  }
#endif
  if (dev == c10::kCPU) {
    msg.device_type = static_cast<int32_t>(kDLCPU);
    msg.device_id = 0;
    msg.data.resize(byte_count);
    return msg;
  }
  throw std::runtime_error(
          "torch_tensor_bridge: unsupported device type " +
          std::to_string(static_cast<int>(dev)));
}

// ---------------------------------------------------------------------------
// DLPack hand-off helpers (framework-agnostic producer API)
// ---------------------------------------------------------------------------

namespace detail
{

/// Context held alive by the DLManagedTensor for the lifetime of the tensor
/// constructed from it. Owns:
///   - (CUDA builds only) a ReadHandle or WriteHandle that keeps the CUDA
///     storage alive and drives event-based synchronization in
///     cuda_buffer_backend, and
///   - stable int64_t storage for DLTensor::shape / DLTensor::strides.
struct BridgeDlCtx
{
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  std::shared_ptr<cuda_buffer_backend::ReadHandle> rh;
  std::shared_ptr<cuda_buffer_backend::WriteHandle> wh;
#endif
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

/// C-style deleter used by every DLManagedTensor produced by the bridge.
/// Called exactly once by the DLPack consumer (e.g. at::fromDLPack, or
/// a framework's from_dlpack) when the imported tensor is destroyed.
inline void bridge_dl_deleter(DLManagedTensor * self)
{
  if (!self) {return;}
  delete static_cast<BridgeDlCtx *>(self->manager_ctx);
  delete self;
}

/// Fill the DLTensor fields of `dlm` from `msg` and `ctx` + resolved pointer
/// and device. Callers set `dlm->manager_ctx` and `dlm->deleter` themselves.
///
/// Note: we bake `msg.byte_offset` into the `data` pointer and set
/// `dl_tensor.byte_offset = 0`. Per the DLPack spec both encodings describe
/// the same tensor, but several DLPack importers (including some versions
/// of torch's `at::fromDLPack`) ignore the `byte_offset` field and read
/// from `data` directly, so baking it in is the portable choice.
inline void fill_dl_tensor(
  DLManagedTensor & dlm,
  const TensorMsg & msg,
  const BridgeDlCtx & ctx,
  void * data_ptr,
  int32_t dev_type)
{
  auto * offset_ptr = static_cast<uint8_t *>(data_ptr) + msg.byte_offset;
  dlm.dl_tensor.data = static_cast<void *>(offset_ptr);
  dlm.dl_tensor.device = DLDevice{static_cast<DLDeviceType>(dev_type), msg.device_id};
  dlm.dl_tensor.ndim = static_cast<int32_t>(ctx.shape.size());
  dlm.dl_tensor.dtype = DLDataType{msg.dtype_code, msg.dtype_bits, msg.dtype_lanes};
  dlm.dl_tensor.shape = const_cast<int64_t *>(ctx.shape.data());
  dlm.dl_tensor.strides = ctx.strides.empty() ?
    nullptr : const_cast<int64_t *>(ctx.strides.data());
  dlm.dl_tensor.byte_offset = 0;
}

}  // namespace detail

#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA

/// Build a DLManagedTensor that references `msg`'s data for reading.
/// The returned pointer is owned by the caller and must be either handed
/// to a DLPack consumer that takes ownership (e.g. `at::fromDLPack`,
/// `jax.dlpack.from_dlpack`, `cupy.from_dlpack`) or released via
/// `dlm->deleter(dlm)`.
///
/// For CUDA-backed `msg.data`, a ReadHandle is acquired on `consumer_stream`
/// (which calls `cudaStreamWaitEvent(consumer_stream, write_event)` to sync
/// against the publisher). For CPU-backed data, `consumer_stream` is ignored.
inline DLManagedTensor * make_dlpack_read(
  const TensorMsg & msg,
  cudaStream_t consumer_stream = nullptr)
{
  auto ctx = std::make_unique<detail::BridgeDlCtx>();
  ctx->shape.assign(msg.shape.begin(), msg.shape.end());
  ctx->strides.assign(msg.strides.begin(), msg.strides.end());

  void * data_ptr = nullptr;
  int32_t dev_type = kDLCPU;

  const std::string & backend = msg.data.get_backend_type();
  if (backend == "cuda") {
    const auto * cuda_impl =
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      msg.data.get_impl());
    if (!cuda_impl) {
      throw std::runtime_error(
              "torch_tensor_bridge::make_dlpack_read: cuda backend but not CudaBufferImpl");
    }
    ctx->rh = std::make_shared<cuda_buffer_backend::ReadHandle>(
      cuda_impl->get_cuda_buffer().get_read_handle(consumer_stream));
    data_ptr = const_cast<void *>(static_cast<const void *>(ctx->rh->get_ptr()));
    dev_type = kDLCUDA;
  } else if (backend == "cpu") {
    (void)consumer_stream;
    data_ptr = const_cast<void *>(static_cast<const void *>(msg.data.data()));
    dev_type = kDLCPU;
  } else {
    throw std::runtime_error(
            "torch_tensor_bridge::make_dlpack_read: unsupported backend '" +
            backend + "'");
  }

  auto * dlm = new DLManagedTensor;
  detail::fill_dl_tensor(*dlm, msg, *ctx, data_ptr, dev_type);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = detail::bridge_dl_deleter;
  return dlm;
}

/// Build a DLManagedTensor that references `msg`'s data for writing.
/// Semantics are the mirror of `make_dlpack_read`: a WriteHandle is
/// acquired for the CUDA path; the publisher's write event is recorded on
/// `consumer_stream` when the imported tensor is destroyed.
inline DLManagedTensor * make_dlpack_write(
  TensorMsg & msg,
  cudaStream_t consumer_stream = nullptr)
{
  auto ctx = std::make_unique<detail::BridgeDlCtx>();
  ctx->shape.assign(msg.shape.begin(), msg.shape.end());
  ctx->strides.assign(msg.strides.begin(), msg.strides.end());

  void * data_ptr = nullptr;
  int32_t dev_type = kDLCPU;

  const std::string & backend = msg.data.get_backend_type();
  if (backend == "cuda") {
    auto * cuda_impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
        msg.data.get_impl()));
    if (!cuda_impl) {
      throw std::runtime_error(
              "torch_tensor_bridge::make_dlpack_write: cuda backend but not CudaBufferImpl");
    }
    cuda_impl->set_stream(consumer_stream);
    ctx->wh = std::make_shared<cuda_buffer_backend::WriteHandle>(
      cuda_impl->get_cuda_buffer().get_write_handle(consumer_stream));
    data_ptr = static_cast<void *>(ctx->wh->get_ptr());
    dev_type = kDLCUDA;
  } else if (backend == "cpu") {
    (void)consumer_stream;
    data_ptr = static_cast<void *>(msg.data.data());
    dev_type = kDLCPU;
  } else {
    throw std::runtime_error(
            "torch_tensor_bridge::make_dlpack_write: unsupported backend '" +
            backend + "'");
  }

  auto * dlm = new DLManagedTensor;
  detail::fill_dl_tensor(*dlm, msg, *ctx, data_ptr, dev_type);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = detail::bridge_dl_deleter;
  return dlm;
}

#else  // TORCH_TENSOR_BRIDGE_HAS_CUDA

/// CPU-only build: only `backend == "cpu"` msgs are supported.
inline DLManagedTensor * make_dlpack_read(const TensorMsg & msg)
{
  if (msg.data.get_backend_type() != "cpu") {
    throw std::runtime_error(
            "torch_tensor_bridge: CUDA not compiled in; cannot handle '" +
            msg.data.get_backend_type() + "' backend");
  }
  auto ctx = std::make_unique<detail::BridgeDlCtx>();
  ctx->shape.assign(msg.shape.begin(), msg.shape.end());
  ctx->strides.assign(msg.strides.begin(), msg.strides.end());

  void * data_ptr = const_cast<void *>(static_cast<const void *>(msg.data.data()));

  auto * dlm = new DLManagedTensor;
  detail::fill_dl_tensor(*dlm, msg, *ctx, data_ptr, kDLCPU);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = detail::bridge_dl_deleter;
  return dlm;
}

inline DLManagedTensor * make_dlpack_write(TensorMsg & msg)
{
  if (msg.data.get_backend_type() != "cpu") {
    throw std::runtime_error(
            "torch_tensor_bridge: CUDA not compiled in; cannot handle '" +
            msg.data.get_backend_type() + "' backend");
  }
  auto ctx = std::make_unique<detail::BridgeDlCtx>();
  ctx->shape.assign(msg.shape.begin(), msg.shape.end());
  ctx->strides.assign(msg.strides.begin(), msg.strides.end());

  void * data_ptr = static_cast<void *>(msg.data.data());

  auto * dlm = new DLManagedTensor;
  detail::fill_dl_tensor(*dlm, msg, *ctx, data_ptr, kDLCPU);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = detail::bridge_dl_deleter;
  return dlm;
}

#endif  // TORCH_TENSOR_BRIDGE_HAS_CUDA

// ---------------------------------------------------------------------------
// Overloaded producer-side entry point + RAII holder
// ---------------------------------------------------------------------------

/// Overloaded producer helper: dispatches to make_dlpack_read or
/// make_dlpack_write based on const-ness of `msg`. Handy for framework
/// bridges that don't care which direction they're in.
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
inline DLManagedTensor * to_dlpack(
  const TensorMsg & msg, cudaStream_t consumer_stream = nullptr)
{
  return make_dlpack_read(msg, consumer_stream);
}

inline DLManagedTensor * to_dlpack(
  TensorMsg & msg, cudaStream_t consumer_stream = nullptr)
{
  return make_dlpack_write(msg, consumer_stream);
}
#else
inline DLManagedTensor * to_dlpack(const TensorMsg & msg)
{
  return make_dlpack_read(msg);
}

inline DLManagedTensor * to_dlpack(TensorMsg & msg)
{
  return make_dlpack_write(msg);
}
#endif

/// RAII wrapper for a DLManagedTensor. Useful when you're not immediately
/// handing the tensor off to a framework's `from_dlpack` (which would take
/// ownership itself). Calling `.release()` hands the raw pointer to such a
/// consumer.
struct DlpackDeleter
{
  void operator()(DLManagedTensor * p) const noexcept
  {
    if (p && p->deleter) {p->deleter(p);}
  }
};

using DlpackPtr = std::unique_ptr<DLManagedTensor, DlpackDeleter>;

#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
inline DlpackPtr to_dlpack_owned(
  const TensorMsg & msg, cudaStream_t consumer_stream = nullptr)
{
  return DlpackPtr{make_dlpack_read(msg, consumer_stream)};
}

inline DlpackPtr to_dlpack_owned(
  TensorMsg & msg, cudaStream_t consumer_stream = nullptr)
{
  return DlpackPtr{make_dlpack_write(msg, consumer_stream)};
}
#else
inline DlpackPtr to_dlpack_owned(const TensorMsg & msg)
{
  return DlpackPtr{make_dlpack_read(msg)};
}

inline DlpackPtr to_dlpack_owned(TensorMsg & msg)
{
  return DlpackPtr{make_dlpack_write(msg)};
}
#endif

// ---------------------------------------------------------------------------
// Torch convenience wrappers on top of DLPack
// ---------------------------------------------------------------------------

namespace detail
{

inline DLManagedTensor * make_dlpack_read_current_stream(const TensorMsg & msg)
{
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  cudaStream_t s = nullptr;
  if (msg.data.get_backend_type() == "cuda") {
    s = current_cuda_stream("from_tensor_msg (read)");
  }
  return make_dlpack_read(msg, s);
#else
  return make_dlpack_read(msg);
#endif
}

inline DLManagedTensor * make_dlpack_write_current_stream(TensorMsg & msg)
{
#ifdef TORCH_TENSOR_BRIDGE_HAS_CUDA
  cudaStream_t s = nullptr;
  if (msg.data.get_backend_type() == "cuda") {
    s = current_cuda_stream("from_tensor_msg (write)");
  }
  return make_dlpack_write(msg, s);
#else
  return make_dlpack_write(msg);
#endif
}

}  // namespace detail

/// Get a writable at::Tensor view over msg.data + msg.byte_offset.
/// The view shares memory with msg.data; the caller must ensure msg outlives
/// the returned tensor. Construction routes through `at::fromDLPack` so the
/// consumer-side DLPack import path is exercised (device / dtype validation
/// happens there).
inline at::Tensor from_tensor_msg(TensorMsg & msg)
{
  if (msg.data.empty()) {return {};}
  DlpackPtr guard{detail::make_dlpack_write_current_stream(msg)};
  at::Tensor t = at::fromDLPack(guard.get());
  (void)guard.release();
  return t;
}

/// Get a read-only at::Tensor from msg.data + msg.byte_offset.
/// \param clone If true (default), returns an independent copy. If false,
/// returns a zero-copy view that keeps a ReadHandle alive via its deleter.
inline at::Tensor from_tensor_msg(const TensorMsg & msg, bool clone = true)
{
  if (msg.data.empty()) {return {};}
  DlpackPtr guard{detail::make_dlpack_read_current_stream(msg)};
  at::Tensor t = at::fromDLPack(guard.get());
  (void)guard.release();
  return clone ? t.clone() : t;
}

namespace detail
{

/// Populate shape / strides / dtype (plus byte_offset = 0) on `msg` from
/// a torch tensor, using `at::toDLPack` to derive the DLPack-form metadata.
/// Device fields on `msg` are intentionally NOT touched: they describe where
/// msg.data physically lives, which is fixed at allocation time and may
/// differ from the source tensor's device.
inline void fill_metadata_via_dlpack(TensorMsg & msg, const at::Tensor & t)
{
  DLManagedTensor * dlm = at::toDLPack(t);
  const DLTensor & dt = dlm->dl_tensor;

  msg.shape.assign(dt.shape, dt.shape + dt.ndim);
  if (dt.strides) {
    msg.strides.assign(dt.strides, dt.strides + dt.ndim);
  } else {
    // DLPack null strides == row-major contiguous; materialize explicit ones.
    msg.strides = contiguous_strides(msg.shape);
  }
  msg.dtype_code = dt.dtype.code;
  msg.dtype_bits = dt.dtype.bits;
  msg.dtype_lanes = dt.dtype.lanes;
  msg.byte_offset = 0;

  if (dlm->deleter) {dlm->deleter(dlm);}
}

}  // namespace detail

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

  detail::fill_metadata_via_dlpack(msg, contig);
}

}  // namespace torch_tensor_bridge

#endif  // TORCH_TENSOR_BRIDGE__TORCH_TENSOR_BRIDGE_HPP_
