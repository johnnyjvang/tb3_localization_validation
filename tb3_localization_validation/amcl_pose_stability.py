#!/usr/bin/env python3

"""
amcl_pose_stability.py

Checks whether AMCL pose remains stable while the robot is stationary.

Usage:
  ros2 run tb3_localization_validation amcl_pose_stability --ros-args -p use_sim_time:=true

-------------------------------------------------------------------------------
IMPORTANT NOTE ON THE FIX (WHY /amcl_pose MAY EXIST BUT THIS NODE GETS NO DATA)
-------------------------------------------------------------------------------

You may see this situation:

    ros2 topic echo /amcl_pose --once

returns a valid message, but your script appears to hang waiting for /amcl_pose,
or times out without collecting samples.

This is usually NOT because AMCL is broken.

The issue is typically ROS 2 QoS compatibility.

WHY THIS HAPPENS:

- /amcl_pose is often published in a latched-style way.
- In ROS 2 terms, that usually means TRANSIENT_LOCAL durability.
- `ros2 topic echo` can often receive the last stored message automatically.
- But a normal subscriber using default QoS may NOT receive that stored message.

So the topic exists and AMCL is publishing, but this node still sees:
    self.latest_msg is None

RESULT:
- wait_for_first_message() may time out
- no samples are collected
- the test fails even though AMCL is actually working

-------------------------------------------------------------------------------
THE FIX
-------------------------------------------------------------------------------

We explicitly subscribe to /amcl_pose using:

    reliability = RELIABLE
    durability = TRANSIENT_LOCAL

This tells ROS 2:

    "Give me the latest stored /amcl_pose message immediately,
     even if it was published before this node started."

That is why the script now works reliably.

-------------------------------------------------------------------------------
WHAT THIS TEST MEASURES
-------------------------------------------------------------------------------

This is NOT a localization accuracy test.

It measures STABILITY of the AMCL pose estimate while the robot is stationary:

- x jitter
- y jitter
- yaw jitter
- x span
- y span
- yaw span
- average covariance in x, y, yaw

This is useful for checking:
- whether AMCL is converged
- whether localization is noisy while stationary
- whether the estimate is steady enough before Nav2 testing

-------------------------------------------------------------------------------
"""

import math
import statistics
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class AmclPoseStability(Node):
    def __init__(self):
        super().__init__('amcl_pose_stability')

        self.declare_parameter('test_duration_sec', 15.0)
        self.declare_parameter('warmup_sec', 3.0)
        self.declare_parameter('wait_for_topic_sec', 15.0)

        self.test_duration = float(self.get_parameter('test_duration_sec').value)
        self.warmup_sec = float(self.get_parameter('warmup_sec').value)
        self.wait_for_topic_sec = float(self.get_parameter('wait_for_topic_sec').value)

        self.latest_msg = None
        self.received_count = 0

        # =====================================================================
        # CRITICAL FIX: AMCL QoS COMPATIBILITY
        # =====================================================================
        #
        # /amcl_pose may use TRANSIENT_LOCAL durability.
        # If we subscribe with default QoS, we may receive nothing even though:
        #
        #   ros2 topic echo /amcl_pose --once
        #
        # works fine.
        #
        # WHY:
        # A default VOLATILE subscriber only gets new messages after it starts.
        # A TRANSIENT_LOCAL subscriber can receive the latest stored AMCL pose.
        #
        # FIX:
        # We explicitly request:
        #   - RELIABLE reliability
        #   - TRANSIENT_LOCAL durability
        #
        amcl_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            amcl_qos
        )

        use_sim_time = False
        if self.has_parameter('use_sim_time'):
            use_sim_time = bool(self.get_parameter('use_sim_time').value)

        self.get_logger().info('Subscribed to /amcl_pose')
        self.get_logger().info(
            f'Parameters: use_sim_time={use_sim_time}, '
            f'test_duration_sec={self.test_duration}, '
            f'warmup_sec={self.warmup_sec}, '
            f'wait_for_topic_sec={self.wait_for_topic_sec}'
        )

    def pose_callback(self, msg):
        self.latest_msg = msg
        self.received_count += 1

        if self.received_count == 1:
            self.get_logger().info('Received first /amcl_pose message.')

    def wait_for_first_message(self):
        self.get_logger().info('Waiting for /amcl_pose messages...')
        start = time.time()

        while rclpy.ok() and self.latest_msg is None:
            rclpy.spin_once(self, timeout_sec=0.2)

            if time.time() - start > self.wait_for_topic_sec:
                self.get_logger().error(
                    f'No /amcl_pose messages received within {self.wait_for_topic_sec:.1f} seconds.'
                )
                return False

        return True

    def collect_samples(self, duration_sec, label='collection'):
        samples_x = []
        samples_y = []
        samples_yaw = []
        cov_xx = []
        cov_yy = []
        cov_yawyaw = []

        start = time.time()
        last_print = start

        while rclpy.ok() and (time.time() - start) < duration_sec:
            rclpy.spin_once(self, timeout_sec=0.2)

            if self.latest_msg is None:
                continue

            pose = self.latest_msg.pose.pose
            cov = self.latest_msg.pose.covariance

            x = pose.position.x
            y = pose.position.y
            yaw = quaternion_to_yaw(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            )

            samples_x.append(x)
            samples_y.append(y)
            samples_yaw.append(yaw)

            # Flattened 6x6 covariance matrix
            # x variance   -> index 0
            # y variance   -> index 7
            # yaw variance -> index 35
            cov_xx.append(cov[0])
            cov_yy.append(cov[7])
            cov_yawyaw.append(cov[35])

            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - start
                self.get_logger().info(
                    f'{label}: {elapsed:.1f}/{duration_sec:.1f} sec, '
                    f'samples={len(samples_x)}, '
                    f'x={x:.4f}, y={y:.4f}, yaw_deg={math.degrees(yaw):.2f}'
                )
                last_print = now

        return samples_x, samples_y, samples_yaw, cov_xx, cov_yy, cov_yawyaw

    def compute_stats(self, xs, ys, yaws, cov_xx, cov_yy, cov_yawyaw):
        if len(xs) < 2:
            return None

        x_mean = statistics.mean(xs)
        y_mean = statistics.mean(ys)

        yaw0 = yaws[0]
        yaw_rel = [wrap_angle(y - yaw0) for y in yaws]
        yaw_mean_rel = statistics.mean(yaw_rel)

        x_std = statistics.stdev(xs)
        y_std = statistics.stdev(ys)
        yaw_std_rad = statistics.stdev(yaw_rel)

        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        yaw_span_rad = max(yaw_rel) - min(yaw_rel)

        avg_cov_xx = statistics.mean(cov_xx)
        avg_cov_yy = statistics.mean(cov_yy)
        avg_cov_yawyaw = statistics.mean(cov_yawyaw)

        return {
            'samples': len(xs),
            'x_mean': x_mean,
            'y_mean': y_mean,
            'yaw_mean_rel_rad': yaw_mean_rel,
            'x_std': x_std,
            'y_std': y_std,
            'yaw_std_rad': yaw_std_rad,
            'x_span': x_span,
            'y_span': y_span,
            'yaw_span_rad': yaw_span_rad,
            'avg_cov_xx': avg_cov_xx,
            'avg_cov_yy': avg_cov_yy,
            'avg_cov_yawyaw': avg_cov_yawyaw,
        }

    def grade_result(self, stats):
        x_std = stats['x_std']
        y_std = stats['y_std']
        yaw_std_deg = math.degrees(stats['yaw_std_rad'])
        x_span = stats['x_span']
        y_span = stats['y_span']
        yaw_span_deg = math.degrees(stats['yaw_span_rad'])

        if (
            x_std <= 0.01 and
            y_std <= 0.01 and
            yaw_std_deg <= 1.5 and
            x_span <= 0.03 and
            y_span <= 0.03 and
            yaw_span_deg <= 4.0
        ):
            return 'PASS'
        elif (
            x_std <= 0.03 and
            y_std <= 0.03 and
            yaw_std_deg <= 4.0 and
            x_span <= 0.08 and
            y_span <= 0.08 and
            yaw_span_deg <= 10.0
        ):
            return 'WARN'
        else:
            return 'FAIL'

    def run_test(self):
        if not self.wait_for_first_message():
            return 1

        self.get_logger().info(
            f'Warmup for {self.warmup_sec:.1f} seconds to let AMCL settle...'
        )
        self.collect_samples(self.warmup_sec, label='warmup')

        self.get_logger().info(
            f'Starting AMCL stationary stability measurement for {self.test_duration:.1f} seconds...'
        )
        xs, ys, yaws, cov_xx, cov_yy, cov_yawyaw = self.collect_samples(
            self.test_duration,
            label='measuring'
        )

        stats = self.compute_stats(xs, ys, yaws, cov_xx, cov_yy, cov_yawyaw)
        if stats is None:
            self.get_logger().error('Not enough /amcl_pose samples collected.')
            return 1

        result = self.grade_result(stats)

        self.get_logger().info('----------------------------------------')
        self.get_logger().info('AMCL Pose Stability Results')
        self.get_logger().info('----------------------------------------')
        self.get_logger().info(f'Result      : {result}')
        self.get_logger().info(f'Samples     : {stats["samples"]}')
        self.get_logger().info(f'X std dev   : {stats["x_std"]:.4f} m')
        self.get_logger().info(f'Y std dev   : {stats["y_std"]:.4f} m')
        self.get_logger().info(f'Yaw std dev : {math.degrees(stats["yaw_std_rad"]):.2f} deg')
        self.get_logger().info(f'X span      : {stats["x_span"]:.4f} m')
        self.get_logger().info(f'Y span      : {stats["y_span"]:.4f} m')
        self.get_logger().info(f'Yaw span    : {math.degrees(stats["yaw_span_rad"]):.2f} deg')
        self.get_logger().info(f'Avg cov x   : {stats["avg_cov_xx"]:.6f}')
        self.get_logger().info(f'Avg cov y   : {stats["avg_cov_yy"]:.6f}')
        self.get_logger().info(f'Avg cov yaw : {stats["avg_cov_yawyaw"]:.6f}')
        self.get_logger().info('----------------------------------------')

        return 0


def main(args=None):
    rclpy.init(args=args)
    node = AmclPoseStability()

    try:
        exit_code = node.run_test()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
        exit_code = 0
    finally:
        node.destroy_node()
        rclpy.shutdown()

    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()