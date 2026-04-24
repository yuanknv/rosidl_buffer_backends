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

#ifndef TORCH_BUFFER_BACKEND__TORCH_BUFFER_BACKEND_HPP_
#define TORCH_BUFFER_BACKEND__TORCH_BUFFER_BACKEND_HPP_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rosidl_buffer_backend/buffer_backend.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "torch_buffer/torch_buffer_impl.hpp"
#include "torch_buffer_backend_msgs/msg/torch_buffer_descriptor.hpp"

namespace torch_buffer_backend
{

/// \brief PyTorch buffer backend plugin for tensor descriptor serialization.
class TorchBufferBackend : public rosidl::BufferBackend
{
public:
  TorchBufferBackend();
  ~TorchBufferBackend() override = default;

  std::string get_backend_type() const override
  {
    return "torch";
  }

  const rosidl_message_type_support_t * get_descriptor_type_support() const override
  {
    return rosidl_typesupport_cpp::get_message_type_support_handle<
      torch_buffer_backend_msgs::msg::TorchBufferDescriptor>();
  }

  std::shared_ptr<void> create_empty_descriptor() const override
  {
    return std::make_shared<torch_buffer_backend_msgs::msg::TorchBufferDescriptor>();
  }

  std::shared_ptr<void> create_descriptor_with_endpoint(
    const void * impl,
    const rmw_topic_endpoint_info_t & endpoint_info) const override
  {
    (void)endpoint_info;
    const auto * torch_impl = dynamic_cast<const torch_buffer_backend::TorchBufferImpl<uint8_t> *>(
      static_cast<const rosidl::BufferImplBase<uint8_t> *>(impl));
    if (!torch_impl) {
      return nullptr;
    }

    auto descriptor = std::make_shared<torch_buffer_backend_msgs::msg::TorchBufferDescriptor>();

    const auto & shape = torch_impl->shape();
    descriptor->shape.assign(shape.begin(), shape.end());

    const auto & strides = torch_impl->strides();
    descriptor->strides.assign(strides.begin(), strides.end());

    descriptor->dtype = torch_impl->dtype();

    if (!torch_impl->get_device_buffer().empty()) {
      descriptor->device_data = torch_impl->get_device_buffer();
    }

    return descriptor;
  }

  std::unique_ptr<void, void (*)(void *)> from_descriptor_with_endpoint(
    const void * descriptor_ptr,
    const rmw_topic_endpoint_info_t & endpoint_info) const override
  {
    (void)endpoint_info;
    const auto * descriptor =
      static_cast<const torch_buffer_backend_msgs::msg::TorchBufferDescriptor *>(
      descriptor_ptr);

    rosidl::Buffer<uint8_t> local_buffer;
    if (!descriptor->device_data.empty()) {
      local_buffer = rosidl::Buffer<uint8_t>(descriptor->device_data);
    }

    auto result = std::make_unique<TorchBufferImpl<uint8_t>>(
      std::move(local_buffer),
      std::vector<int64_t>(descriptor->shape.begin(), descriptor->shape.end()),
      std::vector<int64_t>(descriptor->strides.begin(), descriptor->strides.end()),
      descriptor->dtype);

    return {result.release(), [](void * p) {
        delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
      }};
  }

  std::pair<bool, std::vector<std::set<uint32_t>>> on_discovering_endpoint(
    const rmw_topic_endpoint_info_t & endpoint_info,
    const std::vector<rmw_topic_endpoint_info_t> & existing_endpoints,
    const std::unordered_map<std::string, std::string> & endpoint_supported_backends) override
  {
    (void)endpoint_info;
    (void)existing_endpoints;
    return {
      endpoint_supported_backends.find("torch") != endpoint_supported_backends.end(),
      {}
    };
  }
};

}  // namespace torch_buffer_backend

#endif  // TORCH_BUFFER_BACKEND__TORCH_BUFFER_BACKEND_HPP_
