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

#ifndef TORCH_BUFFER__TORCH_BUFFER_API_HPP_
#define TORCH_BUFFER__TORCH_BUFFER_API_HPP_

#include <c10/core/StreamGuard.h>
#include <rcutils/logging_macros.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rosidl_buffer/buffer.hpp"
#include "torch_buffer/torch_buffer_impl.hpp"
#include "torch_buffer/torch_buffer_utils.hpp"

#if __has_include("cuda_buffer/cuda_buffer_api.hpp")
#include <c10/cuda/CUDAStream.h>
#include "cuda_buffer/cuda_buffer_api.hpp"
#define TORCH_BUFFER_DEVICE_CUDA
#endif

namespace torch_buffer_backend
{

namespace detail
{

inline std::optional<c10::Stream> get_non_default_stream()
{
#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (torch::cuda::is_available()) {
    return c10::cuda::getStreamFromPool();
  }
#endif
  return std::nullopt;
}

inline c10::DeviceType default_device()
{
#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (torch::cuda::is_available()) {
    return c10::kCUDA;
  }
#endif
  return c10::kCPU;
}

template<typename T>
const TorchBufferImpl<T> * get_torch_impl(const rosidl::Buffer<T> & buffer)
{
  if (buffer.get_backend_type() != "torch") {
    throw std::runtime_error(
      "torch_buffer: operation called on non-torch buffer (backend: " +
      buffer.get_backend_type() + ")");
  }
  const auto * impl = buffer.get_impl();
  if (!impl) {
    throw std::runtime_error(
      "torch_buffer: operation called on buffer with null implementation");
  }
  const auto * torch_impl = dynamic_cast<const TorchBufferImpl<T> *>(impl);
  if (!torch_impl) {
    throw std::runtime_error(
      "torch_buffer: failed to cast buffer impl to TorchBufferImpl");
  }
  return torch_impl;
}

template<typename T>
TorchBufferImpl<T> * get_torch_impl(rosidl::Buffer<T> & buffer)
{
  return const_cast<TorchBufferImpl<T> *>(
    get_torch_impl<T>(static_cast<const rosidl::Buffer<T> &>(buffer)));
}

}  // namespace detail

/// \brief RAII guard that sets a non-default CUDA stream for the current scope.
class StreamGuard
{
public:
  StreamGuard()
  : guard_(detail::get_non_default_stream())
  {
  }

private:
  c10::OptionalStreamGuard guard_;
};

/// \brief Create a StreamGuard that activates a non-default CUDA stream.
inline StreamGuard set_stream()
{
  return StreamGuard();
}

/// \brief Allocate a ROS message with a torch-backed buffer.
/// \tparam MsgT ROS message type whose `data` field is a `rosidl::Buffer<uint8_t>`.
/// \param shape Tensor shape.
/// \param dtype Scalar type (e.g. `at::kByte`, `at::kFloat`).
/// \param device Target device (defaults to CUDA if available, otherwise CPU).
template<typename MsgT>
MsgT allocate_msg(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::DeviceType> device = std::nullopt)
{
  c10::DeviceType dev = device.value_or(detail::default_device());

  int64_t numel = 1;
  for (auto s : shape) {
    if (s < 0) {
      throw std::runtime_error(
        "allocate_msg: negative shape dimension (" + std::to_string(s) + ")");
    }
    numel *= s;
  }
  size_t byte_count = static_cast<size_t>(numel) * scalar_type_size(dtype);

  std::vector<int64_t> contiguous_strides(shape.size());
  int64_t stride_val = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    contiguous_strides[i] = stride_val;
    stride_val *= shape[i];
  }

  rosidl::Buffer<uint8_t> device_buffer;

#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (dev == c10::kCUDA) {
    auto cuda_impl = std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(byte_count);
    device_buffer = rosidl::Buffer<uint8_t>(std::move(cuda_impl));
  } else  // NOLINT(readability/braces)
#endif
  if (dev == c10::kCPU) {
    device_buffer.resize(byte_count);
  } else {
    throw std::runtime_error(
      "allocate_msg: unsupported device type " + std::to_string(static_cast<int>(dev)));
  }

  auto torch_impl = std::make_unique<TorchBufferImpl<uint8_t>>(
    std::move(device_buffer), shape, contiguous_strides,
    scalar_type_to_string(dtype));

  MsgT msg;
  msg.data = rosidl::Buffer<uint8_t>(std::move(torch_impl));
  return msg;
}

/// \brief Copy a PyTorch tensor into a pre-allocated torch buffer.
/// The buffer must have been allocated via \c allocate_msg (or otherwise be
/// torch-backed). After the copy, the buffer's tensor metadata (shape,
/// strides, dtype) is updated to match the source tensor.
/// \param buffer Destination buffer (must be torch-backed).
/// \param tensor Source tensor; will be made contiguous if needed.
inline void to_buffer(rosidl::Buffer<uint8_t> & buffer, const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }

  at::Tensor contig = tensor.contiguous();
  size_t byte_count = contig.numel() * contig.element_size();

  auto * torch_impl = detail::get_torch_impl<uint8_t>(buffer);

  if (byte_count > torch_impl->byte_size()) {
    throw std::runtime_error(
      "to_buffer: tensor size (" + std::to_string(byte_count) +
      " bytes) exceeds allocated buffer (" + std::to_string(torch_impl->byte_size()) + " bytes)");
  }

  const std::string & backend = torch_impl->get_device_buffer().get_backend_type();

#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (backend == "cuda") {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto wh = cuda_buffer_backend::from_buffer(
      torch_impl->get_device_buffer(), stream);
    cudaMemcpyKind kind = contig.is_cuda() ?
      cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cuda_buffer_backend::to_buffer(
      contig.data_ptr(), byte_count, wh, stream, kind);
  } else  // NOLINT(readability/braces)
#endif
  if (backend == "cpu") {
    at::Tensor cpu_contig = contig.to(torch::kCPU).contiguous();
    void * dst = torch_impl->get_device_buffer().data();
    std::memcpy(dst, cpu_contig.data_ptr(), byte_count);
  } else {
    throw std::runtime_error("to_buffer: unsupported backend '" + backend + "'");
  }

  torch_impl->set_metadata(
    contig.sizes().vec(), contig.strides().vec(),
    scalar_type_to_string(contig.scalar_type()));
}

namespace detail
{

// Returned tensor borrows memory from impl; caller must ensure
// the owning rosidl::Buffer outlives the tensor.
inline at::Tensor cpu_wrap(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype)
{
  auto * ptr = const_cast<void *>(
    static_cast<const void *>(impl->get_device_buffer().data()));
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  return strides.empty() ?
         torch::from_blob(ptr, shape, opts) :
         torch::from_blob(ptr, shape, strides, opts);
}

#ifdef TORCH_BUFFER_DEVICE_CUDA

inline cudaStream_t get_current_cuda_stream(const char * context)
{
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (stream == nullptr) {
    RCUTILS_LOG_WARN_NAMED(
      "torch_buffer",
      "%s: current CUDA stream is the default stream (nullptr). "
      "Event-based synchronization is disabled; all operations will be synchronous. "
      "Set a non-default stream with c10::cuda::CUDAStreamGuard before calling from_buffer.",
      context);
  }
  return stream;
}

inline at::Tensor cuda_wrap_writable(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype)
{
  auto * cuda_impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      impl->get_device_buffer().get_impl()));
  if (!cuda_impl) {
    throw std::runtime_error("from_buffer (write): device buffer is not a CudaBufferImpl");
  }
  cudaStream_t stream = get_current_cuda_stream("from_buffer (write)");
  cuda_impl->set_stream(stream);
  auto wh = std::make_shared<cuda_buffer_backend::WriteHandle>(
    cuda_impl->get_cuda_buffer().get_write_handle(stream));
  void * ptr = static_cast<void *>(wh->get_ptr());
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
  return strides.empty() ?
         torch::from_blob(ptr, shape, [wh](void *) {}, opts) :
         torch::from_blob(ptr, shape, strides, [wh](void *) {}, opts);
}

inline at::Tensor cuda_wrap_readable(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype,
  bool clone)
{
  const auto * cuda_impl =
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    impl->get_device_buffer().get_impl());
  if (!cuda_impl) {
    throw std::runtime_error("from_buffer (read): device buffer is not a CudaBufferImpl");
  }
  cudaStream_t stream = get_current_cuda_stream("from_buffer (read)");
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
  if (clone) {
    auto rh = cuda_impl->get_cuda_buffer().get_read_handle(stream);
    void * ptr = const_cast<void *>(static_cast<const void *>(rh.get_ptr()));
    at::Tensor view = strides.empty() ?
      torch::from_blob(ptr, shape, opts) :
      torch::from_blob(ptr, shape, strides, opts);
    return view.clone();
  }
  // Zero-copy: keep the ReadHandle alive via the tensor's deleter so the
  // write/read-event machinery stays valid until the tensor is destroyed.
  auto rh = std::make_shared<cuda_buffer_backend::ReadHandle>(
    cuda_impl->get_cuda_buffer().get_read_handle(stream));
  void * ptr = const_cast<void *>(static_cast<const void *>(rh->get_ptr()));
  return strides.empty() ?
         torch::from_blob(ptr, shape, [rh](void *) {}, opts) :
         torch::from_blob(ptr, shape, strides, [rh](void *) {}, opts);
}

#endif  // TORCH_BUFFER_DEVICE_CUDA

template<bool Writable>
inline at::Tensor wrap_impl(
  const TorchBufferImpl<uint8_t> * impl,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides,
  at::ScalarType dtype,
  bool clone)
{
  const std::string & backend = impl->get_device_buffer().get_backend_type();
#ifdef TORCH_BUFFER_DEVICE_CUDA
  if (backend == "cuda") {
    if constexpr (Writable) {return cuda_wrap_writable(impl, shape, strides, dtype);} else {
      return cuda_wrap_readable(impl, shape, strides, dtype, clone);
    }
  }
#endif
  if (backend == "cpu") {
    at::Tensor view = cpu_wrap(impl, shape, strides, dtype);
    if constexpr (!Writable) {
      if (clone) {return view.clone();}
    }
    return view;
  }
  throw std::runtime_error("from_buffer: unsupported backend '" + backend + "'");
}

}  // namespace detail

/// \brief Get a writable tensor view of a torch buffer (uses stored metadata).
inline at::Tensor from_buffer(rosidl::Buffer<uint8_t> & buffer)
{
  if (buffer.empty()) {return {};}
  const auto * impl = detail::get_torch_impl<uint8_t>(buffer);
  at::ScalarType dtype = string_to_scalar_type(impl->dtype());
  return detail::wrap_impl<true>(impl, impl->shape(), impl->strides(), dtype, false);
}

/// \brief Get a read-only tensor of a torch buffer (uses stored metadata).
/// \param clone If true (default), returns an independent copy that the
/// caller can safely mutate. If false, returns a tensor that directly views
/// the buffer's memory (zero-copy); the returned tensor owns a
/// \c ReadHandle via its deleter, so the source buffer stays alive until
/// the tensor is destroyed. The caller must treat a non-cloned tensor as
/// read-only; mutating it corrupts the shared buffer memory.
inline at::Tensor from_buffer(const rosidl::Buffer<uint8_t> & buffer, bool clone = true)
{
  if (buffer.empty()) {return {};}
  const auto * impl = detail::get_torch_impl<uint8_t>(buffer);
  at::ScalarType dtype = string_to_scalar_type(impl->dtype());
  return detail::wrap_impl<false>(impl, impl->shape(), impl->strides(), dtype, clone);
}

/// \brief Get a writable tensor view with explicit shape, strides, and dtype.
inline at::Tensor from_buffer(
  rosidl::Buffer<uint8_t> & buffer,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides = {},
  at::ScalarType dtype = at::kByte)
{
  if (buffer.empty()) {return {};}
  const auto * impl = detail::get_torch_impl<uint8_t>(buffer);
  return detail::wrap_impl<true>(impl, shape, strides, dtype, false);
}

/// \brief Get a read-only tensor with explicit shape, strides, and dtype.
/// \param clone See the simpler overload; defaults to true.
inline at::Tensor from_buffer(
  const rosidl::Buffer<uint8_t> & buffer,
  const std::vector<int64_t> & shape,
  const std::vector<int64_t> & strides = {},
  at::ScalarType dtype = at::kByte,
  bool clone = true)
{
  if (buffer.empty()) {return {};}
  const auto * impl = detail::get_torch_impl<uint8_t>(buffer);
  return detail::wrap_impl<false>(impl, shape, strides, dtype, clone);
}

}  // namespace torch_buffer_backend

#endif  // TORCH_BUFFER__TORCH_BUFFER_API_HPP_
