#!/usr/bin/env python3

"""
yaw_alignment_check.py

Compare yaw between:
- AMCL (global)
- Odom (local)

Ensures both frames remain aligned.

-------------------------------------------------------------------------------
IMPORTANT NOTE ON QoS (THIS FIXES THE "NO AMCL DATA" ISSUE)
-------------------------------------------------------------------------------

You may encounter a situation where:

    ros2 topic echo /amcl_pose --once

works, but this node prints:

    Waiting for topics: /amcl_pose | amcl_msgs=0

This is NOT because AMCL is broken.

This is a ROS 2 QoS (Quality of Service) issue.

WHY THIS HAPPENS:

- /amcl_pose is often published with TRANSIENT_LOCAL durability.
- This means it behaves like a "latched" topic (it stores the last message).
- Tools like `ros2 topic echo` automatically receive that stored message.

BUT:

- A normal ROS2 subscription uses VOLATILE durability (default).
- That means it ONLY receives messages published AFTER it subscribes.
- If AMCL is not actively publishing (or publishing slowly), your node gets NOTHING.

RESULT:
- Your node never receives /amcl_pose
- It appears like AMCL is not working
- But it actually is

THE FIX:

We explicitly request:

    durability = TRANSIENT_LOCAL
    reliability = RELIABLE

This tells ROS2:

    "Give me the most recent stored AMCL message immediately"

This is why the node suddenly starts working.

RULE OF THUMB:

- Use TRANSIENT_LOCAL for:
    /amcl_pose, /map (latched-style topics)

- Use SENSOR DATA QoS for:
    /odom, /scan, /imu (high-rate streaming topics)

-------------------------------------------------------------------------------
"""

import math
import time
import statistics

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)

from .result_utils import append_result


def get_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


class YawAlignmentCheck(Node):
    def __init__(self):
        super().__init__('yaw_alignment_check')

        self.duration = 15.0
        self.start_time = time.time()

        self.amcl = None
        self.odom = None
        self.errors = []
        self.amcl_count = 0
        self.odom_count = 0
        self.last_status_second = -1

        # ===============================
        # AMCL QoS FIX (CRITICAL)
        # ===============================
        #
        # We use TRANSIENT_LOCAL so we can receive the last stored AMCL pose.
        # Without this, the node may never receive any AMCL messages.
        #
        amcl_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_cb,
            amcl_qos
        )

        # ===============================
        # ODOM QoS
        # ===============================
        #
        # Odom is a high-frequency streaming topic.
        # Best practice is to use sensor-data QoS.
        #
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_cb,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(0.5, self.run)

        self.get_logger().info('========== YAW ALIGNMENT CHECK ==========')

    def amcl_cb(self, msg):
        self.amcl = msg
        self.amcl_count += 1

    def odom_cb(self, msg):
        self.odom = msg
        self.odom_count += 1

    def run(self):
        elapsed = time.time() - self.start_time
        current_second = int(elapsed)

        if elapsed < self.duration:
            if self.amcl is None or self.odom is None:
                if current_second != self.last_status_second:
                    self.last_status_second = current_second

                    missing = []
                    if self.amcl is None:
                        missing.append('/amcl_pose')
                    if self.odom is None:
                        missing.append('/odom')

                    self.get_logger().info(
                        f'[INFO] Waiting for topics: {", ".join(missing)} '
                        f'| amcl_msgs={self.amcl_count}, odom_msgs={self.odom_count}'
                    )
                return

            yaw_amcl = get_yaw(self.amcl.pose.pose.orientation)
            yaw_odom = get_yaw(self.odom.pose.pose.orientation)

            raw_error_deg = math.degrees(yaw_amcl - yaw_odom)
            error_deg = abs(normalize_angle_deg(raw_error_deg))
            self.errors.append(error_deg)

            self.get_logger().info(
                f'[RUNNING] yaw_amcl={math.degrees(yaw_amcl):.2f} deg, '
                f'yaw_odom={math.degrees(yaw_odom):.2f} deg, '
                f'error={error_deg:.2f} deg'
            )
            return

        # ===== ANALYSIS =====
        if len(self.errors) < 5:
            append_result(
                'yaw_alignment_check',
                'FAIL',
                '0.00 deg',
                f'not enough data; amcl_msgs={self.amcl_count}, odom_msgs={self.odom_count}'
            )
            self.get_logger().error('[FAIL] Not enough data collected')
            self.shutdown()
            return

        avg_error = statistics.mean(self.errors)
        min_error = min(self.errors)
        max_error = max(self.errors)

        if avg_error < 5.0:
            result = 'PASS'
        elif avg_error < 10.0:
            result = 'WARN'
        else:
            result = 'FAIL'

        append_result(
            'yaw_alignment_check',
            result,
            f'{avg_error:.2f} deg',
            f'min={min_error:.2f} deg, max={max_error:.2f} deg, samples={len(self.errors)}'
        )

        self.get_logger().info(f'[{result}] Avg yaw error = {avg_error:.2f} deg')
        self.shutdown()

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = YawAlignmentCheck()

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