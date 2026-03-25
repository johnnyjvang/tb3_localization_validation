#!/usr/bin/env python3

"""
covariance_monitor.py

Monitor AMCL covariance over time to ensure localization confidence is stable.

Checks:
- Covariance in X, Y, and Yaw from /amcl_pose
- Ensures values are within expected bounds

Why this test matters:
- Lower covariance generally indicates AMCL is more confident
- Very high covariance can indicate poor convergence or poor localization quality

Outputs:
- PASS / WARN / FAIL

Writes:
- results.csv via append_result(...)
"""

import time
import statistics

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from .result_utils import append_result


class CovarianceMonitor(Node):
    def __init__(self):
        super().__init__('covariance_monitor')

        # Test timing
        self.duration_sec = 15.0
        self.start_time = time.time()

        # Latest AMCL message
        self.latest_msg = None
        self.received_count = 0

        # Storage for covariance samples
        self.cov_x = []
        self.cov_y = []
        self.cov_yaw = []

        # AMCL publisher may require a more compatible QoS
        amcl_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscribe to AMCL pose
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            amcl_qos
        )

        # Periodic timer to run the test logic
        self.timer = self.create_timer(0.5, self.run)

        use_sim_time = False
        if self.has_parameter('use_sim_time'):
            use_sim_time = bool(self.get_parameter('use_sim_time').value)

        self.get_logger().info('========== COVARIANCE MONITOR ==========')
        self.get_logger().info(f'[INFO] use_sim_time = {use_sim_time}')
        self.get_logger().info('[INFO] Subscribed to /amcl_pose')

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        """Store the latest AMCL pose message."""
        self.latest_msg = msg
        self.received_count += 1

        if self.received_count == 1:
            self.get_logger().info('[INFO] Received first /amcl_pose message')

    def run(self):
        """
        Collect covariance samples while the timer is active.
        After duration expires, compute averages and write results.
        """
        elapsed = time.time() - self.start_time

        # During collection period
        if elapsed < self.duration_sec:
            if self.latest_msg is None:
                self.get_logger().info('[INFO] Waiting for /amcl_pose...')
                return

            cov = self.latest_msg.pose.covariance

            # 6x6 flattened covariance matrix indices
            # x   = 0
            # y   = 7
            # yaw = 35
            cov_x_val = cov[0]
            cov_y_val = cov[7]
            cov_yaw_val = cov[35]

            self.cov_x.append(cov_x_val)
            self.cov_y.append(cov_y_val)
            self.cov_yaw.append(cov_yaw_val)

            self.get_logger().info(
                f'[RUNNING] t={elapsed:.1f}s, '
                f'cov_x={cov_x_val:.4f}, cov_y={cov_y_val:.4f}, cov_yaw={cov_yaw_val:.4f}'
            )
            return

        # ===== ANALYSIS =====
        if len(self.cov_x) < 5:
            self.get_logger().error('[FAIL] Not enough data collected')
            append_result(
                'covariance_monitor',
                'FAIL',
                '0.0',
                'not enough /amcl_pose samples'
            )
            self.shutdown()
            return

        avg_x = statistics.mean(self.cov_x)
        avg_y = statistics.mean(self.cov_y)
        avg_yaw = statistics.mean(self.cov_yaw)

        min_x = min(self.cov_x)
        min_y = min(self.cov_y)
        min_yaw = min(self.cov_yaw)

        max_x = max(self.cov_x)
        max_y = max(self.cov_y)
        max_yaw = max(self.cov_yaw)

        self.get_logger().info(f'[INFO] Samples      : {len(self.cov_x)}')
        self.get_logger().info(f'[INFO] Avg Cov X   : {avg_x:.4f}')
        self.get_logger().info(f'[INFO] Avg Cov Y   : {avg_y:.4f}')
        self.get_logger().info(f'[INFO] Avg Cov Yaw : {avg_yaw:.4f}')
        self.get_logger().info(f'[INFO] Min Cov X   : {min_x:.4f}')
        self.get_logger().info(f'[INFO] Min Cov Y   : {min_y:.4f}')
        self.get_logger().info(f'[INFO] Min Cov Yaw : {min_yaw:.4f}')
        self.get_logger().info(f'[INFO] Max Cov X   : {max_x:.4f}')
        self.get_logger().info(f'[INFO] Max Cov Y   : {max_y:.4f}')
        self.get_logger().info(f'[INFO] Max Cov Yaw : {max_yaw:.4f}')

        # Starter thresholds
        # These can be adjusted later after observing real/sim behavior
        if avg_x < 0.05 and avg_y < 0.05 and avg_yaw < 0.10:
            result = 'PASS'
        elif avg_x < 0.15 and avg_y < 0.15 and avg_yaw < 0.30:
            result = 'WARN'
        else:
            result = 'FAIL'

        append_result(
            'covariance_monitor',
            result,
            f'{avg_x:.4f}',
            f'y={avg_y:.4f}, yaw={avg_yaw:.4f}'
        )

        self.get_logger().info(f'[{result}] Covariance monitor complete')
        self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        self.get_logger().info('Shutting down covariance_monitor')
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