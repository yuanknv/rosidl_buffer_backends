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
#include <torch/torch.h>

#include <vector>

#include "torch_tensor_bridge/torch_tensor_bridge.hpp"

using torch_tensor_bridge::TensorMsg;
using torch_tensor_bridge::allocate_tensor;
using torch_tensor_bridge::from_tensor_msg;
using torch_tensor_bridge::to_tensor_msg;
using torch_tensor_bridge::dlpack::kDLCPU;
using torch_tensor_bridge::dlpack::kDLUInt;
using torch_tensor_bridge::dlpack::kDLInt;
using torch_tensor_bridge::dlpack::kDLFloat;

TEST(TorchTensorBridge, AllocateCpuTensorPopulatesDlpackMetadata)
{
  TensorMsg msg = allocate_tensor({2, 3, 4}, at::kFloat, c10::kCPU);

  ASSERT_EQ(msg.shape.size(), 3u);
  EXPECT_EQ(msg.shape[0], 2);
  EXPECT_EQ(msg.shape[1], 3);
  EXPECT_EQ(msg.shape[2], 4);

  ASSERT_EQ(msg.strides.size(), 3u);
  EXPECT_EQ(msg.strides[0], 12);
  EXPECT_EQ(msg.strides[1], 4);
  EXPECT_EQ(msg.strides[2], 1);

  EXPECT_EQ(msg.dtype_code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(msg.dtype_bits, 32u);
  EXPECT_EQ(msg.dtype_lanes, 1u);

  EXPECT_EQ(msg.device_type, static_cast<int32_t>(kDLCPU));
  EXPECT_EQ(msg.device_id, 0);

  EXPECT_EQ(msg.byte_offset, 0u);
  EXPECT_EQ(msg.data.size(), 2u * 3u * 4u * sizeof(float));
  EXPECT_EQ(msg.data.get_backend_type(), "cpu");
}

TEST(TorchTensorBridge, ByteDtypeRoundTripsThroughDlpackTriple)
{
  TensorMsg msg = allocate_tensor({5}, at::kByte, c10::kCPU);
  EXPECT_EQ(msg.dtype_code, static_cast<uint8_t>(kDLUInt));
  EXPECT_EQ(msg.dtype_bits, 8u);
  EXPECT_EQ(msg.dtype_lanes, 1u);
}

TEST(TorchTensorBridge, Int32DtypeRoundTripsThroughDlpackTriple)
{
  TensorMsg msg = allocate_tensor({4}, at::kInt, c10::kCPU);
  EXPECT_EQ(msg.dtype_code, static_cast<uint8_t>(kDLInt));
  EXPECT_EQ(msg.dtype_bits, 32u);
}

TEST(TorchTensorBridge, WriteThenReadRoundTrip)
{
  TensorMsg msg = allocate_tensor({4}, at::kInt, c10::kCPU);

  {
    at::Tensor t = from_tensor_msg(msg);
    ASSERT_TRUE(t.defined());
    EXPECT_EQ(t.sizes(), (std::vector<int64_t>{4}));
    EXPECT_EQ(t.scalar_type(), at::kInt);
    t.copy_(torch::tensor({10, 20, 30, 40}, at::kInt));
  }

  at::Tensor v = from_tensor_msg(
    const_cast<const TensorMsg &>(msg), /*clone=*/false);
  ASSERT_TRUE(v.defined());
  ASSERT_EQ(v.numel(), 4);
  auto * p = v.data_ptr<int32_t>();
  EXPECT_EQ(p[0], 10);
  EXPECT_EQ(p[1], 20);
  EXPECT_EQ(p[2], 30);
  EXPECT_EQ(p[3], 40);
}

TEST(TorchTensorBridge, ToTensorMsgCopiesAndUpdatesMetadata)
{
  TensorMsg msg = allocate_tensor({16}, at::kFloat, c10::kCPU);

  at::Tensor src = torch::arange(0, 6, at::kFloat).reshape({2, 3});
  to_tensor_msg(msg, src);

  ASSERT_EQ(msg.shape.size(), 2u);
  EXPECT_EQ(msg.shape[0], 2);
  EXPECT_EQ(msg.shape[1], 3);
  EXPECT_EQ(msg.dtype_code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(msg.dtype_bits, 32u);
  EXPECT_EQ(msg.byte_offset, 0u);

  at::Tensor round = from_tensor_msg(
    const_cast<const TensorMsg &>(msg), /*clone=*/true);
  ASSERT_EQ(round.numel(), 6);
  EXPECT_TRUE(torch::equal(round.flatten(), src.flatten()));
}

TEST(TorchTensorBridge, ByteOffsetSelectsSubregionOfStorage)
{
  // Allocate 16 ints but publish only a 4-int view starting at index 4.
  TensorMsg msg = allocate_tensor({16}, at::kInt, c10::kCPU);
  {
    at::Tensor full = from_tensor_msg(msg);
    for (int i = 0; i < 16; ++i) {
      full.index_put_({i}, i * 100);
    }
  }

  msg.shape = {4};
  msg.strides = {1};
  msg.byte_offset = 4 * sizeof(int32_t);

  at::Tensor view = from_tensor_msg(
    const_cast<const TensorMsg &>(msg), /*clone=*/false);
  ASSERT_EQ(view.numel(), 4);
  auto * p = view.data_ptr<int32_t>();
  EXPECT_EQ(p[0], 400);
  EXPECT_EQ(p[1], 500);
  EXPECT_EQ(p[2], 600);
  EXPECT_EQ(p[3], 700);
}

TEST(TorchTensorBridge, ToTensorMsgRejectsOversizedTensor)
{
  TensorMsg msg = allocate_tensor({4}, at::kByte, c10::kCPU);
  at::Tensor big = torch::zeros({128}, at::kByte);
  EXPECT_THROW(to_tensor_msg(msg, big), std::runtime_error);
}

TEST(TorchTensorBridge, EmptyDataReturnsUndefinedTensor)
{
  TensorMsg msg;
  EXPECT_FALSE(from_tensor_msg(const_cast<const TensorMsg &>(msg)).defined());
  EXPECT_FALSE(from_tensor_msg(msg).defined());
}

TEST(TorchTensorBridge, DtypeConversionRejectsUnsupportedTriple)
{
  using torch_tensor_bridge::DLDataType;
  using torch_tensor_bridge::scalar_from_dl_dtype;
  EXPECT_THROW(scalar_from_dl_dtype(DLDataType{kDLFloat, 128, 1}), std::runtime_error);
  EXPECT_THROW(scalar_from_dl_dtype(DLDataType{kDLFloat, 32, 4}), std::runtime_error);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
