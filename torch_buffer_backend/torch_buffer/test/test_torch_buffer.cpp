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

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <vector>

#include "torch_buffer/torch_buffer_impl.hpp"
#include "torch_buffer/torch_buffer_utils.hpp"
#include "rosidl_buffer/buffer.hpp"

using torch_buffer_backend::TorchBufferImpl;

TEST(TorchBufferAccess, CpuImplBasics)
{
  rosidl::Buffer<uint8_t> device_buf(10);
  for (size_t i = 0; i < 10; ++i) {
    device_buf[i] = static_cast<uint8_t>(i);
  }

  auto impl = std::make_unique<TorchBufferImpl<uint8_t>>(
    std::move(device_buf), std::vector<int64_t>{2, 5},
    std::vector<int64_t>{5, 1}, "Byte");

  EXPECT_EQ(impl->size(), 10u);
  EXPECT_EQ(impl->byte_size(), 10u);
  EXPECT_EQ(impl->device_type(), "cpu");
  EXPECT_EQ(impl->dtype(), "Byte");
  ASSERT_EQ(impl->shape().size(), 2u);
  EXPECT_EQ(impl->shape()[0], 2);
  EXPECT_EQ(impl->shape()[1], 5);
  ASSERT_EQ(impl->strides().size(), 2u);
  EXPECT_EQ(impl->strides()[0], 5);
  EXPECT_EQ(impl->strides()[1], 1);
  EXPECT_NE(impl->get_device_impl(), nullptr);
}

TEST(TorchBufferAccess, FloatImplSizeConversion)
{
  rosidl::Buffer<uint8_t> device_buf(20);
  auto impl = std::make_unique<TorchBufferImpl<float>>(
    std::move(device_buf), std::vector<int64_t>{5},
    std::vector<int64_t>{1}, "Float");

  EXPECT_EQ(impl->byte_size(), 20u);
  EXPECT_EQ(impl->size(), 5u);
}

TEST(TorchBufferAccess, Clone)
{
  rosidl::Buffer<uint8_t> device_buf(8);
  for (size_t i = 0; i < 8; ++i) {
    device_buf[i] = static_cast<uint8_t>(i * 10);
  }

  auto impl = std::make_unique<TorchBufferImpl<uint8_t>>(
    std::move(device_buf), std::vector<int64_t>{8},
    std::vector<int64_t>{1}, "Byte");

  auto cloned = impl->clone();
  ASSERT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->size(), 8u);

  auto * cloned_torch = static_cast<TorchBufferImpl<uint8_t> *>(cloned.get());
  EXPECT_EQ(cloned_torch->dtype(), "Byte");
  EXPECT_EQ(cloned_torch->shape()[0], 8);
}

TEST(TorchBufferAccess, ToCpu)
{
  rosidl::Buffer<uint8_t> device_buf(6);
  for (size_t i = 0; i < 6; ++i) {
    device_buf[i] = static_cast<uint8_t>(i);
  }

  auto impl = std::make_unique<TorchBufferImpl<uint8_t>>(
    std::move(device_buf), std::vector<int64_t>{2, 3},
    std::vector<int64_t>{3, 1}, "Byte");

  std::unique_ptr<rosidl::BufferImplBase<uint8_t>> cpu_copy = impl->to_cpu();
  ASSERT_NE(cpu_copy, nullptr);
  EXPECT_EQ(cpu_copy->size(), 6u);

  auto * cpu_impl = static_cast<rosidl::CpuBufferImpl<uint8_t> *>(cpu_copy.get());
  const std::vector<uint8_t> & storage = cpu_impl->get_storage();
  ASSERT_EQ(storage.size(), 6u);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(storage[i], static_cast<uint8_t>(i));
  }
}

TEST(TorchBufferAccess, EmptyImpl)
{
  auto impl = std::make_unique<TorchBufferImpl<uint8_t>>();
  EXPECT_EQ(impl->size(), 0u);
  EXPECT_EQ(impl->device_type(), "cpu");
}

TEST(TorchBufferAccess, Clear)
{
  rosidl::Buffer<uint8_t> device_buf(10);
  auto impl = std::make_unique<TorchBufferImpl<uint8_t>>(
    std::move(device_buf), std::vector<int64_t>{10},
    std::vector<int64_t>{1}, "Byte");

  EXPECT_EQ(impl->size(), 10u);
  impl->clear();
  EXPECT_EQ(impl->size(), 0u);
  EXPECT_TRUE(impl->shape().empty());
  EXPECT_TRUE(impl->strides().empty());
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
