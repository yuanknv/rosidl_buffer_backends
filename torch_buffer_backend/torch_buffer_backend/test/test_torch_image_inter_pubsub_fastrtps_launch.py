#!/usr/bin/env python3
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

import time
import unittest

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from std_msgs.msg import Bool, Float64, UInt32


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """Generate launch description for Torch image inter-process test with FastRTPS."""
    publisher_node = Node(
        package='torch_buffer_backend',
        executable='torch_image_publisher_node',
        name='torch_image_publisher',
        output='screen',
        parameters=[{
            'max_publish_count': 0,
            'publish_rate_ms': 100,
            'image_width': 1920,
            'image_height': 1080,
        }],
    )

    subscriber_node = Node(
        package='torch_buffer_backend',
        executable='torch_image_subscriber_node',
        name='torch_image_subscriber',
        output='screen',
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        subscriber_node,
        TimerAction(period=2.0, actions=[
            publisher_node,
            launch_testing.actions.ReadyToTest(),
        ]),
    ])


class TestTorchImageInterPubSubFastRTPS(unittest.TestCase):
    """Test Torch Buffer-based Image pub/sub across separate processes over FastRTPS."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_torch_image_inter_pubsub_fastrtps')
        self.publisher_count = 0
        self.subscriber_count = 0
        self.validation_passed = True
        self.latencies = []

        self.node.create_subscription(
            UInt32, 'publisher_count', self._pub_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'subscriber_count', self._sub_count_cb, 10)
        self.node.create_subscription(
            Bool, 'validation_result', self._validation_cb, 10)
        self.node.create_subscription(
            Float64, 'latency_ms', self._latency_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _pub_count_cb(self, msg):
        self.publisher_count = msg.data

    def _sub_count_cb(self, msg):
        self.subscriber_count = msg.data

    def _validation_cb(self, msg):
        self.validation_passed = msg.data

    def _latency_cb(self, msg):
        self.latencies.append(msg.data)

    def _spin_until(self, target_count=5, timeout_sec=15.0):
        start = time.time()
        while self.subscriber_count < target_count and time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.subscriber_count >= target_count

    def test_inter_process_pubsub(self):
        """Test torch buffer-based inter-process image pub/sub over FastRTPS."""
        success = self._spin_until(target_count=5, timeout_sec=15.0)

        self.assertTrue(
            success,
            f'Failed to receive 5 messages within timeout (inter-process). '
            f'Received: {self.subscriber_count}')
        self.assertGreaterEqual(
            self.publisher_count, 5,
            f'Publisher should have sent at least 5 messages. '
            f'Sent: {self.publisher_count}')
        self.assertTrue(
            self.validation_passed, 'Image validation failed (inter-process)')
        self.assertLessEqual(
            abs(self.publisher_count - self.subscriber_count), 5,
            f'Publisher count ({self.publisher_count}) and subscriber count '
            f'({self.subscriber_count}) differ by more than 5')


@launch_testing.post_shutdown_test()
class TestTorchImageInterPubSubFastRTPSShutdown(unittest.TestCase):
    """Test proper shutdown of inter-process nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, 1, -2, -6, -15],
        )
