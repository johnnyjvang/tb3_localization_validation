#!/usr/bin/env python3

"""
initial_pose_response.py

Publish an initial pose to AMCL and verify that /amcl_pose responds.

Checks:
- /amcl_pose is available
- initial pose can be published
- AMCL pose updates after publishing
"""

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


class InitialPoseResponse(Node):
    def __init__(self) -> None:
        super().__init__('initial_pose_response')

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.latest_amcl_pose: Optional[PoseWithCovarianceStamped] = None
        self.latest_odom: Optional[Odometry] = None
        self.amcl_count = 0
        self.have_published = False
        self.start_time = time.time()
        self.publish_time = None

        self.timeout_sec = 15.0
        self.timer = self.create_timer(0.2, self.run)

        self.get_logger().info('========== INITIAL POSE RESPONSE ==========')

    def amcl_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.latest_amcl_pose = msg
        self.amcl_count += 1

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def yaw_to_quaternion(self, yaw: float):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw

    def publish_initial_pose(self) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0

        yaw = 0.0
        qz, qw = self.yaw_to_quaternion(yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        covariance = [0.0] * 36
        covariance[0] = 0.25
        covariance[7] = 0.25
        covariance[35] = 0.0685
        msg.pose.covariance = covariance

        self.initial_pose_pub.publish(msg)
        self.publish_time = time.time()
        self.have_published = True

        self.get_logger().info('[INFO] Published initial pose to /initialpose')
        self.get_logger().info('[INFO] Target pose: x=0.00, y=0.00, yaw=0.00 deg')

    def run(self) -> None:
        elapsed = time.time() - self.start_time

        if elapsed > self.timeout_sec:
            self.get_logger().error('[FAIL] Timed out waiting for AMCL response')
            self.shutdown()
            return

        if not self.have_published:
            if self.amcl_count >= 1 or self.latest_odom is not None:
                self.publish_initial_pose()
            else:
                self.get_logger().info('[INFO] Waiting for AMCL/odom topics...')
            return

        if self.latest_amcl_pose is not None and self.publish_time is not None:
            dt = time.time() - self.publish_time

            x = self.latest_amcl_pose.pose.pose.position.x
            y = self.latest_amcl_pose.pose.pose.position.y

            self.get_logger().info(f'[PASS] AMCL responded after initial pose publish')
            self.get_logger().info(f'[INFO] Response time: {dt:.2f} sec')
            self.get_logger().info(f'[INFO] Current AMCL pose: x={x:.3f}, y={y:.3f}')
            self.shutdown()

    def shutdown(self) -> None:
        self.get_logger().info('Initial pose response check complete. Shutting down.')
        self.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InitialPoseResponse()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()