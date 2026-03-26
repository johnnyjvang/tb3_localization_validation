#!/usr/bin/env python3

"""
covariance_monitor.py

Monitor AMCL covariance stability from /amcl_pose.

Checks:
- x covariance
- y covariance
- yaw covariance

This helps confirm that localization uncertainty remains reasonably low and stable.

-------------------------------------------------------------------------------
IMPORTANT NOTE ON THE FIX (WHY THIS SCRIPT MAY FAIL EVEN WHEN /amcl_pose EXISTS)
-------------------------------------------------------------------------------

You may encounter a situation where:

    ros2 topic echo /amcl_pose --once

works fine, but this node prints something like:

    [INFO] Waiting for /amcl_pose...

or eventually fails with:

    [FAIL] Not enough data collected

This is usually NOT because AMCL is broken.

The real issue is often ROS 2 QoS compatibility.

WHY THIS HAPPENS:

- /amcl_pose is commonly published in a way that behaves like a "latched" topic.
- In ROS 2 terms, that usually means TRANSIENT_LOCAL durability.
- Tools like `ros2 topic echo` can often receive the stored last message.
- But a normal Python subscriber using default QoS may NOT receive that stored AMCL message.

So the node appears to get "zero data" even though the topic clearly exists.

-------------------------------------------------------------------------------
THE FIX
-------------------------------------------------------------------------------

We explicitly create an AMCL subscription QoS profile with:

    reliability = RELIABLE
    durability = TRANSIENT_LOCAL

This tells ROS 2:

    "Give me the most recent stored AMCL pose, even if it was published before
     this node started."

Without this, the script may wait forever for /amcl_pose or fail to collect
enough covariance samples.

-------------------------------------------------------------------------------
WHAT THIS TEST ACTUALLY MEASURES
-------------------------------------------------------------------------------

This test reads the covariance values from PoseWithCovarianceStamped:

    covariance[0]   -> x covariance
    covariance[7]   -> y covariance
    covariance[35]  -> yaw covariance

These are the correct indices in the flattened 6x6 covariance matrix for:
- x position uncertainty
- y position uncertainty
- yaw uncertainty

The script records these over time and reports the average values.

This is NOT a full localization accuracy test.
It is a localization uncertainty / confidence monitoring test.

-------------------------------------------------------------------------------
"""

import time
import statistics

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from .result_utils import append_result


class CovarianceMonitor(Node):
    def __init__(self):
        super().__init__('covariance_monitor')

        self.duration = 10.0
        self.start_time = time.time()

        self.latest_msg = None
        self.cov_x = []
        self.cov_y = []
        self.cov_yaw = []

        # =====================================================================
        # CRITICAL FIX: AMCL QoS COMPATIBILITY
        # =====================================================================
        #
        # Default subscription QoS may fail to receive /amcl_pose even though:
        #   ros2 topic echo /amcl_pose --once
        # still works.
        #
        # WHY:
        # /amcl_pose may use TRANSIENT_LOCAL durability, which means the last
        # message is stored and delivered only to subscribers that request it
        # with compatible QoS.
        #
        # FIX:
        # We explicitly request:
        #   - RELIABLE reliability
        #   - TRANSIENT_LOCAL durability
        #
        # This allows the node to receive the most recent AMCL pose immediately.
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
            self.amcl_callback,
            amcl_qos
        )

        self.timer = self.create_timer(0.5, self.run)

        self.get_logger().info('========== COVARIANCE MONITOR ==========')

    def amcl_callback(self, msg):
        self.latest_msg = msg

    def run(self):
        elapsed = time.time() - self.start_time

        if elapsed < self.duration:
            if self.latest_msg is None:
                self.get_logger().info('[INFO] Waiting for /amcl_pose...')
                return

            cov = self.latest_msg.pose.covariance

            # =================================================================
            # COVARIANCE INDEXING
            # =================================================================
            #
            # PoseWithCovariance stores a flattened 6x6 covariance matrix.
            #
            # The indices used here are:
            #   cov[0]   -> variance of x
            #   cov[7]   -> variance of y
            #   cov[35]  -> variance of yaw
            #
            # These are the standard indices for:
            #   x, y, yaw
            #
            self.cov_x.append(cov[0])
            self.cov_y.append(cov[7])
            self.cov_yaw.append(cov[35])

            self.get_logger().info(
                f'[RUNNING] cov_x={cov[0]:.4f}, cov_y={cov[7]:.4f}, cov_yaw={cov[35]:.4f}'
            )
            return

        # ===== ANALYSIS =====
        if len(self.cov_x) < 5:
            append_result(
                'covariance_monitor',
                'FAIL',
                '0.0000',
                'not enough data collected'
            )
            self.get_logger().error('[FAIL] Not enough data collected')
            self.shutdown()
            return

        avg_x = statistics.mean(self.cov_x)
        avg_y = statistics.mean(self.cov_y)
        avg_yaw = statistics.mean(self.cov_yaw)

        # =====================================================================
        # SIMPLE PASS / WARN / FAIL THRESHOLDS
        # =====================================================================
        #
        # These are heuristic thresholds for basic localization confidence.
        # Lower covariance generally means AMCL is more confident.
        #
        if avg_x < 0.1 and avg_y < 0.1 and avg_yaw < 0.1:
            result = 'PASS'
        elif avg_x < 0.3 and avg_y < 0.3 and avg_yaw < 0.3:
            result = 'WARN'
        else:
            result = 'FAIL'

        append_result(
            'covariance_monitor',
            result,
            f'x={avg_x:.4f}, y={avg_y:.4f}, yaw={avg_yaw:.4f}',
            ''
        )

        self.get_logger().info(
            f'[{result}] Avg covariance x={avg_x:.4f}, y={avg_y:.4f}, yaw={avg_yaw:.4f}'
        )
        self.shutdown()

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CovarianceMonitor()

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