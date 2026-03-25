#!/usr/bin/env python3

"""
amcl_pose_stability.py

Measure AMCL pose stability while the robot remains stationary.

Checks:
- /amcl_pose is being published
- pose jitter in x/y/yaw over a fixed sample window
"""

import math
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped


class AMCLPoseStability(Node):
    def __init__(self) -> None:
        super().__init__('amcl_pose_stability')

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )

        self.samples: List[Tuple[float, float, float]] = []
        self.start_time = time.time()
        self.last_progress_second = -1

        self.sample_duration = 10.0
        self.timeout_sec = 20.0

        self.timer = self.create_timer(0.2, self.run)

        self.get_logger().info('========== AMCL POSE STABILITY ==========')

    def quaternion_to_yaw(self, z: float, w: float) -> float:
        return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)

    def amcl_callback(self, msg: PoseWithCovarianceStamped) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        yaw = self.quaternion_to_yaw(z, w)

        self.samples.append((x, y, yaw))

    def run(self) -> None:
        elapsed = time.time() - self.start_time
        current_second = int(elapsed)

        if current_second != self.last_progress_second:
            self.last_progress_second = current_second
            self.get_logger().info(
                f'[INFO] Recording stationary AMCL pose... {elapsed:.1f}/{self.sample_duration:.1f} sec '
                f'| samples: {len(self.samples)}'
            )

        if elapsed >= self.timeout_sec and len(self.samples) < 2:
            self.get_logger().error('[FAIL] Timed out waiting for enough /amcl_pose samples')
            self.shutdown()
            return

        if elapsed >= self.sample_duration and len(self.samples) >= 2:
            self.analyze()

    def analyze(self) -> None:
        xs = [s[0] for s in self.samples]
        ys = [s[1] for s in self.samples]
        yaws = [math.degrees(s[2]) for s in self.samples]

        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        yaw_span = max(yaws) - min(yaws)

        self.get_logger().info('')
        self.get_logger().info('========== ANALYSIS ==========')
        self.get_logger().info(f'[INFO] Samples collected: {len(self.samples)}')
        self.get_logger().info(f'[INFO] X span: {x_span:.4f} m')
        self.get_logger().info(f'[INFO] Y span: {y_span:.4f} m')
        self.get_logger().info(f'[INFO] Yaw span: {yaw_span:.4f} deg')

        if x_span <= 0.05 and y_span <= 0.05 and yaw_span <= 5.0:
            self.get_logger().info('[PASS] AMCL pose appears stable while stationary')
        else:
            self.get_logger().warn('[WARN] AMCL pose jitter is larger than expected')

        self.shutdown()

    def shutdown(self) -> None:
        self.get_logger().info('AMCL pose stability check complete. Shutting down.')
        self.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AMCLPoseStability()

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