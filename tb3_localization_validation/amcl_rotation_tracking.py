#!/usr/bin/env python3

"""
amcl_rotation_tracking.py

Command an in-place rotation and compare yaw tracking between:
- AMCL (/amcl_pose)
- Odom (/odom)

Goal:
- Verify AMCL updates during rotation
- Check whether AMCL yaw change roughly agrees with odom yaw change

-------------------------------------------------------------------------------
IMPORTANT NOTE ON QoS
-------------------------------------------------------------------------------

/amcl_pose may exist and be visible with:

    ros2 topic echo /amcl_pose --once

while a normal subscriber still receives nothing.

WHY:
- /amcl_pose often behaves like a latched topic
- In ROS 2 this usually means TRANSIENT_LOCAL durability
- Default VOLATILE subscribers may miss the stored AMCL message

FIX:
- Subscribe to /amcl_pose with:
    reliability = RELIABLE
    durability = TRANSIENT_LOCAL

For /odom:
- use qos_profile_sensor_data because it is a high-rate streaming topic

-------------------------------------------------------------------------------
WHAT THIS TEST MEASURES
-------------------------------------------------------------------------------

This is not a perfect localization accuracy test.

It measures whether:
- AMCL yaw changes when the robot rotates
- Odom yaw changes when the robot rotates
- The AMCL yaw change is reasonably close to odom yaw change

This is useful before Nav2 because it tells you whether localization is
responsive during turning.
"""

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)

from .result_utils import append_result


# ===== Topics =====
AMCL_TOPIC = '/amcl_pose'
ODOM_TOPIC = '/odom'
CMD_VEL_TOPIC = '/cmd_vel'

# ===== Timing =====
CONTROL_PERIOD = 0.05
PROGRESS_PERIOD = 1.0
STARTUP_WAIT_TIMEOUT = 15.0
MAX_TEST_TIME = 30.0
SETTLE_TIME = 1.0

# ===== Motion =====
ROTATE_SPEED = 0.4       # rad/s
ROTATE_DURATION = 3.0    # sec

# ===== Thresholds =====
MIN_ODOM_ROTATION_DEG = 20.0
PASS_DIFF_DEG = 10.0
WARN_DIFF_DEG = 20.0


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class AMCLRotationTracking(Node):
    def __init__(self) -> None:
        super().__init__('amcl_rotation_tracking')

        # =====================================================================
        # CRITICAL FIX: QoS for /amcl_pose
        # =====================================================================
        amcl_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.amcl_msg: Optional[PoseWithCovarianceStamped] = None
        self.odom_msg: Optional[Odometry] = None
        self.amcl_count = 0
        self.odom_count = 0

        self.create_subscription(
            PoseWithCovarianceStamped,
            AMCL_TOPIC,
            self.amcl_cb,
            amcl_qos
        )

        self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.odom_cb,
            qos_profile_sensor_data
        )

        self.cmd_pub = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)

        self.start_time = time.time()
        self.phase = 'wait_for_ready'
        self.phase_start_time = time.time()
        self.finish_time = None
        self.done = False

        self.start_amcl_yaw = None
        self.start_odom_yaw = None
        self.end_amcl_yaw = None
        self.end_odom_yaw = None

        self.timer = self.create_timer(CONTROL_PERIOD, self.loop)
        self.progress_timer = self.create_timer(PROGRESS_PERIOD, self.progress_update)

        self.get_logger().info('========== AMCL ROTATION TRACKING ==========')
        self.get_logger().info(f'AMCL topic: {AMCL_TOPIC}')
        self.get_logger().info(f'Odom topic: {ODOM_TOPIC}')
        self.get_logger().info(f'Command topic: {CMD_VEL_TOPIC}')
        self.get_logger().info(f'Rotate speed: {ROTATE_SPEED:.2f} rad/s')
        self.get_logger().info(f'Rotate duration: {ROTATE_DURATION:.2f} sec')

    def amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self.amcl_msg = msg
        self.amcl_count += 1

    def odom_cb(self, msg: Odometry) -> None:
        self.odom_msg = msg
        self.odom_count += 1

    def publish_cmd(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        cmd = TwistStamped()
        cmd.twist.linear.x = linear_x
        cmd.twist.angular.z = angular_z
        self.cmd_pub.publish(cmd)

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)

    def have_required_topics(self) -> bool:
        return self.amcl_msg is not None and self.odom_msg is not None

    def get_amcl_yaw(self) -> Optional[float]:
        if self.amcl_msg is None:
            return None
        q = self.amcl_msg.pose.pose.orientation
        return quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def get_odom_yaw(self) -> Optional[float]:
        if self.odom_msg is None:
            return None
        q = self.odom_msg.pose.pose.orientation
        return quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def transition(self, new_phase: str) -> None:
        self.phase = new_phase
        self.phase_start_time = time.time()
        self.get_logger().info(f'[INFO] Transition -> {new_phase}')

    def progress_update(self) -> None:
        if self.done:
            return

        elapsed = time.time() - self.start_time
        phase_elapsed = time.time() - self.phase_start_time

        self.get_logger().info(
            f'[Progress] {elapsed:.1f}s / {MAX_TEST_TIME:.1f}s | '
            f'phase: {self.phase} | '
            f'phase_elapsed: {phase_elapsed:.1f}s | '
            f'amcl_msgs: {self.amcl_count} | odom_msgs: {self.odom_count}'
        )

    def finish_and_exit(self, status: str, measurement: str, notes: str) -> None:
        self.stop_robot()

        append_result(
            'amcl_rotation_tracking',
            status,
            measurement,
            notes
        )

        if status == 'PASS':
            self.get_logger().info(f'Result: {status}')
        elif status == 'WARN':
            self.get_logger().warn(f'Result: {status}')
        else:
            self.get_logger().error(f'Result: {status}')

        self.done = True
        self.finish_time = time.time()

    def analyze(self) -> None:
        self.get_logger().info('')
        self.get_logger().info('========== ANALYSIS ==========')

        if (
            self.start_amcl_yaw is None or
            self.start_odom_yaw is None or
            self.end_amcl_yaw is None or
            self.end_odom_yaw is None
        ):
            self.finish_and_exit(
                'FAIL',
                'missing yaw snapshots',
                'could not record required AMCL/odom yaw values'
            )
            return

        amcl_delta = normalize_angle(self.end_amcl_yaw - self.start_amcl_yaw)
        odom_delta = normalize_angle(self.end_odom_yaw - self.start_odom_yaw)

        amcl_delta_deg = math.degrees(amcl_delta)
        odom_delta_deg = math.degrees(odom_delta)
        diff_deg = abs(amcl_delta_deg - odom_delta_deg)

        self.get_logger().info(f'[INFO] AMCL yaw change: {amcl_delta_deg:.2f} deg')
        self.get_logger().info(f'[INFO] Odom yaw change: {odom_delta_deg:.2f} deg')
        self.get_logger().info(f'[INFO] Absolute difference: {diff_deg:.2f} deg')

        if abs(odom_delta_deg) < MIN_ODOM_ROTATION_DEG:
            self.finish_and_exit(
                'FAIL',
                f'odom rotation too small ({odom_delta_deg:.2f} deg)',
                'robot did not rotate enough for meaningful comparison'
            )
            return

        if diff_deg <= PASS_DIFF_DEG:
            status = 'PASS'
        elif diff_deg <= WARN_DIFF_DEG:
            status = 'WARN'
        else:
            status = 'FAIL'

        measurement = (
            f'amcl={amcl_delta_deg:.2f} deg | '
            f'odom={odom_delta_deg:.2f} deg | '
            f'diff={diff_deg:.2f} deg'
        )
        notes = (
            f'rotate_speed={ROTATE_SPEED:.2f} rad/s, '
            f'rotate_duration={ROTATE_DURATION:.2f} sec'
        )

        if status == 'PASS':
            self.get_logger().info('[PASS] AMCL rotation tracking looks good')
        elif status == 'WARN':
            self.get_logger().warn('[WARN] AMCL rotation tracking is usable but not ideal')
        else:
            self.get_logger().error('[FAIL] AMCL rotation tracking differs too much from odom')

        self.finish_and_exit(status, measurement, notes)

    def loop(self) -> None:
        now = time.time()
        elapsed = now - self.start_time
        phase_elapsed = now - self.phase_start_time

        if self.done:
            if now - self.finish_time > 0.5:
                self.get_logger().info('Exiting amcl_rotation_tracking')
                rclpy.shutdown()
            return

        if elapsed > MAX_TEST_TIME:
            self.finish_and_exit(
                'FAIL',
                'timed out',
                'rotation tracking test timed out'
            )
            return

        if self.phase == 'wait_for_ready':
            self.stop_robot()

            if self.have_required_topics():
                self.start_amcl_yaw = self.get_amcl_yaw()
                self.start_odom_yaw = self.get_odom_yaw()

                if self.start_amcl_yaw is None or self.start_odom_yaw is None:
                    return

                self.get_logger().info(
                    f'[INFO] Start AMCL yaw: {math.degrees(self.start_amcl_yaw):.2f} deg'
                )
                self.get_logger().info(
                    f'[INFO] Start Odom yaw: {math.degrees(self.start_odom_yaw):.2f} deg'
                )
                self.transition('rotating')
                return

            if phase_elapsed >= STARTUP_WAIT_TIMEOUT:
                self.finish_and_exit(
                    'FAIL',
                    'startup readiness timeout',
                    f'timed out waiting for topics; amcl_msgs={self.amcl_count}, odom_msgs={self.odom_count}'
                )
                return
            return

        if self.phase == 'rotating':
            if phase_elapsed < ROTATE_DURATION:
                self.publish_cmd(0.0, ROTATE_SPEED)
            else:
                self.stop_robot()
                self.transition('settling')
            return

        if self.phase == 'settling':
            self.stop_robot()
            if phase_elapsed >= SETTLE_TIME:
                self.end_amcl_yaw = self.get_amcl_yaw()
                self.end_odom_yaw = self.get_odom_yaw()

                self.get_logger().info(
                    f'[INFO] End AMCL yaw: {math.degrees(self.end_amcl_yaw):.2f} deg'
                )
                self.get_logger().info(
                    f'[INFO] End Odom yaw: {math.degrees(self.end_odom_yaw):.2f} deg'
                )

                self.analyze()
            return


def main(args=None):
    rclpy.init(args=args)
    node = AMCLRotationTracking()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        if rclpy.ok():
            node.stop_robot()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()