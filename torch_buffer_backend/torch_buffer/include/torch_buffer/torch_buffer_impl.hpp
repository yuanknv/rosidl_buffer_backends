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

#ifndef TORCH_BUFFER__TORCH_BUFFER_IMPL_HPP_
#define TORCH_BUFFER__TORCH_BUFFER_IMPL_HPP_

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rosidl_buffer/buffer.hpp"
#include "rosidl_buffer/buffer_impl_base.hpp"
#include "rosidl_buffer/cpu_buffer_impl.hpp"

namespace torch_buffer_backend
{

/// \brief Device buffer wrapper with tensor metadata (shape, strides, dtype).
template<typename T>
class TorchBufferImpl : public rosidl::BufferImplBase<T>
{
public:
  TorchBufferImpl() {}

  TorchBufferImpl(
    rosidl::Buffer<uint8_t> && device_buffer,
    const std::vector<int64_t> & shape,
    const std::vector<int64_t> & strides,
    const std::string & dtype)
  : device_buffer_(std::move(device_buffer)),
    shape_(shape),
    strides_(strides),
    dtype_(dtype)
  {
  }

  TorchBufferImpl(const TorchBufferImpl &) = delete;
  TorchBufferImpl & operator=(const TorchBufferImpl &) = delete;
  TorchBufferImpl(TorchBufferImpl &&) = default;
  TorchBufferImpl & operator=(TorchBufferImpl &&) = default;

  std::string get_backend_type() const override {return "torch";}

  size_t size() const override
  {
    return device_buffer_.size() / sizeof(T);
  }

  size_t byte_size() const
  {
    return device_buffer_.size();
  }

  void resize(size_t n)
  {
    device_buffer_.resize(n * sizeof(T));
  }

  void clear()
  {
    device_buffer_.clear();
    shape_.clear();
    strides_.clear();
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> to_cpu() const override
  {
    if (device_buffer_.empty()) {return std::make_unique<rosidl::CpuBufferImpl<T>>();}
    std::vector<uint8_t> cpu_vec = device_buffer_.to_vector();
    auto cpu_impl = std::make_unique<rosidl::CpuBufferImpl<T>>();
    cpu_impl->get_storage().resize(cpu_vec.size() / sizeof(T));
    std::memcpy(cpu_impl->get_storage().data(), cpu_vec.data(), cpu_vec.size());
    return cpu_impl;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> clone() const override
  {
    if (device_buffer_.empty()) {
      return std::make_unique<TorchBufferImpl<T>>();
    }
    rosidl::Buffer<uint8_t> cloned_buffer(device_buffer_);
    return std::make_unique<TorchBufferImpl<T>>(
      std::move(cloned_buffer), shape_, strides_, dtype_);
  }

  std::string device_type() const {return device_buffer_.get_backend_type();}
  const std::vector<int64_t> & shape() const {return shape_;}
  const std::vector<int64_t> & strides() const {return strides_;}
  const std::string & dtype() const {return dtype_;}

  const rosidl::BufferImplBase<uint8_t> * get_device_impl() const
  {
    return device_buffer_.get_impl();
  }

  rosidl::Buffer<uint8_t> & get_device_buffer()
  {
    return device_buffer_;
  }

  const rosidl::Buffer<uint8_t> & get_device_buffer() const
  {
    return device_buffer_;
  }

  void set_metadata(
    const std::vector<int64_t> & shape,
    const std::vector<int64_t> & strides,
    const std::string & dtype)
  {
    shape_ = shape;
    strides_ = strides;
    dtype_ = dtype;
  }

private:
  rosidl::Buffer<uint8_t> device_buffer_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  std::string dtype_;
};

}  // namespace torch_buffer_backend

#endif  // TORCH_BUFFER__TORCH_BUFFER_IMPL_HPP_
