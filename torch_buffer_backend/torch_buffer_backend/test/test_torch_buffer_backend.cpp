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
#include <memory>
#include <vector>

#include "torch_buffer/torch_buffer_api.hpp"
#include "rosidl_buffer/buffer.hpp"
#include "sensor_msgs/msg/image.hpp"

static bool has_cuda()
{
  if (!torch::cuda::is_available()) {return false;}
  try {
    torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
      {1}, torch::kByte, c10::kCUDA);
    return true;
  } catch (...) {
    return false;
  }
}

TEST(TorchBackend, AllocateAndWrite)
{
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {10}, torch::kByte);

  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  output.copy_(
    torch::arange(10, torch::TensorOptions().dtype(torch::kByte).device(output.device())) * 2);

  const rosidl::Buffer<uint8_t> & cbuf = msg.data;
  at::Tensor read_back = torch_buffer_backend::from_buffer(cbuf);
  EXPECT_EQ(read_back.numel(), 10);
  at::Tensor cpu_read_back = read_back.cpu();
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(cpu_read_back[i].item<uint8_t>(), static_cast<uint8_t>(i * 2));
  }
}

TEST(TorchBackend, MetadataPreserved)
{
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {2, 3}, torch::kByte);

  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  output.copy_(
    torch::arange(6, torch::TensorOptions().dtype(torch::kByte).device(output.device()))
    .view({2, 3}));

  const rosidl::Buffer<uint8_t> & cbuf = msg.data;
  at::Tensor recovered = torch_buffer_backend::from_buffer(cbuf);
  EXPECT_EQ(recovered.dim(), 2);
  EXPECT_EQ(recovered.size(0), 2);
  EXPECT_EQ(recovered.size(1), 3);
  at::Tensor cpu_recovered = recovered.cpu().view({-1});
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(cpu_recovered[i].item<uint8_t>(), static_cast<uint8_t>(i));
  }
}

TEST(TorchBackend, FloatType)
{
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {5}, torch::kFloat);

  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  output.copy_(
    torch::arange(5, torch::TensorOptions().dtype(torch::kFloat).device(output.device())) * 0.5f);

  const rosidl::Buffer<uint8_t> & cbuf = msg.data;
  at::Tensor tensor = torch_buffer_backend::from_buffer(cbuf);
  EXPECT_EQ(tensor.numel(), 5);
  EXPECT_EQ(tensor.dtype(), torch::kFloat);
  at::Tensor cpu_tensor = tensor.cpu();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(cpu_tensor[i].item<float>(), static_cast<float>(i) * 0.5f);
  }
}

TEST(TorchBackend, EmptyBuffer)
{
  rosidl::Buffer<uint8_t> empty_buffer;
  at::Tensor tensor = torch_buffer_backend::from_buffer(empty_buffer);
  EXPECT_EQ(tensor.numel(), 0);
}

TEST(TorchBackend, BackendType)
{
  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {5}, torch::kByte);
  EXPECT_EQ(msg.data.get_backend_type(), "torch");
}

TEST(TorchBackend, DoubleWriteHandleThrows)
{
  if (!has_cuda()) {GTEST_SKIP() << "No CUDA device";}
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {4}, torch::kByte, c10::kCUDA);

  at::Tensor first = torch_buffer_backend::from_buffer(msg.data);
  EXPECT_THROW(
    torch_buffer_backend::from_buffer(msg.data),
    std::runtime_error);
}

TEST(TorchBackend, FloatRoundTrip)
{
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {100}, torch::kFloat);

  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  output.copy_(
    torch::arange(100, torch::TensorOptions().dtype(torch::kFloat).device(output.device())) +
      1.0f);

  const rosidl::Buffer<uint8_t> & cbuf = msg.data;
  at::Tensor recovered = torch_buffer_backend::from_buffer(cbuf);
  at::Tensor cpu_recovered = recovered.cpu();
  for (int64_t i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(cpu_recovered[i].item<float>(), static_cast<float>(i) + 1.0f);
  }
}

TEST(TorchBackend, ToCpuFallback)
{
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {8}, torch::kByte);

  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  output.copy_(
    torch::arange(8, torch::TensorOptions().dtype(torch::kByte).device(output.device())));

  std::vector<uint8_t> vec = msg.data.to_vector();
  ASSERT_EQ(vec.size(), 8u);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(vec[i], static_cast<uint8_t>(i));
  }
}

TEST(TorchBackendOverride, ExplicitCpu)
{
  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {4}, torch::kByte, c10::kCPU);

  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  EXPECT_FALSE(output.is_cuda());
  EXPECT_EQ(output.numel(), 4);
}

TEST(TorchBackendOverride, ExplicitCuda)
{
  if (!has_cuda()) {GTEST_SKIP() << "No CUDA device";}
  torch_buffer_backend::StreamGuard guard = torch_buffer_backend::set_stream();

  sensor_msgs::msg::Image msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
    {4}, torch::kByte, c10::kCUDA);

  at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
  EXPECT_TRUE(output.is_cuda());
  EXPECT_EQ(output.numel(), 4);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
