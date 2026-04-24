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

#include <torch/torch.h>
#include <cstring>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64.hpp"
#include "torch_buffer/torch_buffer_api.hpp"

class TorchImageSubscriber : public rclcpp::Node
{
public:
  explicit TorchImageSubscriber(const rclcpp::NodeOptions & options)
  : Node("torch_image_subscriber", options),
    received_count_(0),
    validation_passed_(true)
  {
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.acceptable_buffer_backends = "any";
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "test_torch_image", 10,
      std::bind(&TorchImageSubscriber::image_callback, this, std::placeholders::_1),
      sub_opts);

    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>("subscriber_count", 10);
    validation_publisher_ = this->create_publisher<std_msgs::msg::Bool>("validation_result", 10);
    backend_validation_publisher_ =
      this->create_publisher<std_msgs::msg::Bool>("backend_validation", 10);
    content_validation_publisher_ =
      this->create_publisher<std_msgs::msg::Bool>("content_validation", 10);
    metadata_validation_publisher_ =
      this->create_publisher<std_msgs::msg::Bool>("metadata_validation", 10);
    latency_publisher_ = this->create_publisher<std_msgs::msg::Float64>("latency_ms", 10);

    RCLCPP_INFO(this->get_logger(), "Torch image subscriber started");
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    double latency_ms = (this->now() - msg->header.stamp).seconds() * 1000.0;
    received_count_++;
    bool metadata_valid = true;
    bool backend_valid = true;
    bool content_valid = true;

    size_t expected_size = msg->width * msg->height * 3;

    if (msg->encoding != "rgb8") {
      RCLCPP_ERROR(this->get_logger(), "Wrong encoding: %s", msg->encoding.c_str());
      metadata_valid = false;
    }

    if (msg->data.size() != expected_size) {
      RCLCPP_ERROR(this->get_logger(), "Wrong data size: %zu (expected %zu)",
                   msg->data.size(), expected_size);
      metadata_valid = false;
    }

    const std::string backend_type = msg->data.get_backend_type();
    if (backend_type != "torch") {
      RCLCPP_ERROR(this->get_logger(),
        "Wrong backend type: %s (expected: torch)", backend_type.c_str());
      backend_valid = false;
    }

    if (backend_type == "torch" && !msg->data.empty()) {
      size_t buffer_byte_size = msg->data.size();
      const rosidl::Buffer<uint8_t> & data = msg->data;
      at::Tensor tensor = torch_buffer_backend::from_buffer(data);

      at::Tensor cpu_tensor = tensor.contiguous().cpu();
      if (cpu_tensor.numel() != static_cast<int64_t>(buffer_byte_size)) {
        RCLCPP_ERROR(this->get_logger(),
          "Tensor element count mismatch: %ld vs %zu",
          cpu_tensor.numel(), buffer_byte_size);
        content_valid = false;
      }

      if (content_valid && cpu_tensor.numel() > 0) {
        at::Tensor flat = cpu_tensor.view({-1});
        const uint8_t * ptr = flat.data_ptr<uint8_t>();
        uint8_t expected_val = ptr[0];
        for (int64_t i = 1; i < flat.numel(); ++i) {
          if (ptr[i] != expected_val) {
            RCLCPP_ERROR(this->get_logger(),
              "Content corruption at byte %ld: expected 0x%02x, got 0x%02x",
              i, expected_val, ptr[i]);
            content_valid = false;
            break;
          }
        }
      }
    }

    bool msg_valid = metadata_valid && backend_valid && content_valid;

    std_msgs::msg::Float64 latency_msg;
    latency_msg.data = latency_ms;
    latency_publisher_->publish(latency_msg);

    validation_passed_ = validation_passed_ && msg_valid;

    std_msgs::msg::UInt32 count_msg;
    count_msg.data = received_count_;
    count_publisher_->publish(count_msg);

    std_msgs::msg::Bool validation_msg;
    validation_msg.data = validation_passed_;
    validation_publisher_->publish(validation_msg);

    std_msgs::msg::Bool backend_msg;
    backend_msg.data = backend_valid;
    backend_validation_publisher_->publish(backend_msg);

    std_msgs::msg::Bool content_msg;
    content_msg.data = content_valid;
    content_validation_publisher_->publish(content_msg);

    std_msgs::msg::Bool metadata_msg;
    metadata_msg.data = metadata_valid;
    metadata_validation_publisher_->publish(metadata_msg);

    if (!msg_valid) {
      RCLCPP_ERROR(this->get_logger(), "Received INVALID image #%u", received_count_);
    } else {
      RCLCPP_INFO(this->get_logger(),
                  "Received image #%u (%ux%u, %zu bytes, "
                  "backend: %s, latency: %.3f ms)",
                  received_count_, msg->width, msg->height,
                  msg->data.size(), backend_type.c_str(), latency_ms);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr validation_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr backend_validation_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr content_validation_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr metadata_validation_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr latency_publisher_;
  uint32_t received_count_;
  bool validation_passed_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(TorchImageSubscriber)
