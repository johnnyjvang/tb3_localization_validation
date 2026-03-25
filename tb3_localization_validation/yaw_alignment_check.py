#!/usr/bin/env python3

"""
yaw_alignment_check.py

Compare yaw between:
- AMCL (global)
- Odom (local)

Ensures both frames remain aligned.
"""

import math
import time
import statistics

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from .result_utils import write_result


def get_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class YawAlignmentCheck(Node):
    def __init__(self):
        super().__init__('yaw_alignment_check')

        self.duration = 15.0
        self.start_time = time.time()

        self.amcl = None
        self.odom = None
        self.errors = []

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_cb,
            10
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_cb,
            10
        )

        self.timer = self.create_timer(0.5, self.run)

        self.get_logger().info('========== YAW ALIGNMENT CHECK ==========')

    def amcl_cb(self, msg):
        self.amcl = msg

    def odom_cb(self, msg):
        self.odom = msg

    def run(self):
        elapsed = time.time() - self.start_time

        if elapsed < self.duration:
            if self.amcl is None or self.odom is None:
                self.get_logger().info('[INFO] Waiting for AMCL/Odom...')
                return

            yaw_amcl = get_yaw(self.amcl.pose.pose.orientation)
            yaw_odom = get_yaw(self.odom.pose.pose.orientation)

            error = abs(math.degrees(yaw_amcl - yaw_odom))
            self.errors.append(error)

            self.get_logger().info(f'[RUNNING] yaw error = {error:.2f} deg')
            return

        # ===== ANALYSIS =====
        if len(self.errors) < 5:
            write_result('yaw_alignment_check', 'FAIL', 0.0, 'no data')
            self.shutdown()
            return

        avg_error = statistics.mean(self.errors)

        if avg_error < 5:
            result = 'PASS'
        elif avg_error < 10:
            result = 'WARN'
        else:
            result = 'FAIL'

        write_result('yaw_alignment_check', result, avg_error, '')

        self.get_logger().info(f'[{result}] Avg yaw error = {avg_error:.2f} deg')
        self.shutdown()

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = YawAlignmentCheck()
    rclpy.spin(node)


if __name__ == '__main__':
    main()