#!/usr/bin/env python3

"""
amcl_relocalization_test.py

Simple relocalization test:
1. Wait for /amcl_pose and /odom
2. Record start AMCL + odom pose
3. Publish an intentionally wrong /initialpose
4. Move the robot
5. Record end AMCL + odom pose
6. Compare AMCL motion vs odom motion

Purpose:
- Verify AMCL recovers enough to track translation + rotation after being
  intentionally given a wrong initial pose.

"""

import math
import time
from typing import Optional, Tuple

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
INITIALPOSE_TOPIC = '/initialpose'
CMD_VEL_TOPIC = '/cmd_vel'

# ===== Timing =====
CONTROL_PERIOD = 0.05
PROGRESS_PERIOD = 1.0
STARTUP_WAIT_TIMEOUT = 15.0
POST_WRONG_POSE_SETTLE = 2.0
SETTLE_TIME = 3.0
MAX_TEST_TIME = 50.0

# ===== Wrong pose offsets =====
WRONG_OFFSET_X = 1.0
WRONG_OFFSET_Y = 0.5
WRONG_OFFSET_YAW_DEG = 30.0

# ===== Motion sequence =====
ROTATE_SPEED = 0.35
ROTATE_DURATION = 2.0

FORWARD_SPEED = 0.08
FORWARD_DURATION = 5.0

# ===== Thresholds =====
MIN_ODOM_DISTANCE_M = 0.20
PASS_DIST_DIFF_M = 0.15
WARN_DIST_DIFF_M = 0.30

PASS_YAW_DIFF_DEG = 15.0
WARN_YAW_DIFF_DEG = 30.0


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw_rad: float) -> Tuple[float, float]:
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return qz, qw


class AMCLRelocalizationTest(Node):
    def __init__(self) -> None:
        super().__init__('amcl_relocalization_test')

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

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            INITIALPOSE_TOPIC,
            10
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            CMD_VEL_TOPIC,
            10
        )

        self.start_time = time.time()
        self.phase = 'wait_for_ready'
        self.phase_start_time = time.time()
        self.finish_time = None
        self.done = False

        self.start_amcl_pose = None
        self.start_odom_pose = None
        self.end_amcl_pose = None
        self.end_odom_pose = None

        self.start_amcl_count = 0
        self.end_amcl_count = 0

        self.timer = self.create_timer(CONTROL_PERIOD, self.loop)
        self.progress_timer = self.create_timer(PROGRESS_PERIOD, self.progress_update)

        self.get_logger().info('========== AMCL RELOCALIZATION TEST ==========')
        self.get_logger().info(f'AMCL topic: {AMCL_TOPIC}')
        self.get_logger().info(f'Odom topic: {ODOM_TOPIC}')
        self.get_logger().info(f'Initial pose topic: {INITIALPOSE_TOPIC}')
        self.get_logger().info(f'Command topic: {CMD_VEL_TOPIC}')

    def amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self.amcl_msg = msg
        self.amcl_count += 1

    def odom_cb(self, msg: Odometry) -> None:
        self.odom_msg = msg
        self.odom_count += 1

    def have_required_topics(self) -> bool:
        return self.amcl_msg is not None and self.odom_msg is not None

    def get_amcl_pose(self) -> Optional[Tuple[float, float, float]]:
        if self.amcl_msg is None:
            return None
        p = self.amcl_msg.pose.pose.position
        q = self.amcl_msg.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        return (p.x, p.y, yaw)

    def get_odom_pose(self) -> Optional[Tuple[float, float, float]]:
        if self.odom_msg is None:
            return None
        p = self.odom_msg.pose.pose.position
        q = self.odom_msg.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        return (p.x, p.y, yaw)

    def publish_cmd(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        cmd = TwistStamped()
        cmd.twist.linear.x = linear_x
        cmd.twist.angular.z = angular_z
        self.cmd_pub.publish(cmd)

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)

    def publish_wrong_initial_pose(self) -> None:
        current = self.get_amcl_pose()
        if current is None:
            return

        x, y, yaw = current
        wrong_x = x + WRONG_OFFSET_X
        wrong_y = y + WRONG_OFFSET_Y
        wrong_yaw = yaw + math.radians(WRONG_OFFSET_YAW_DEG)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = wrong_x
        msg.pose.pose.position.y = wrong_y
        msg.pose.pose.position.z = 0.0

        qz, qw = yaw_to_quaternion(wrong_yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        cov = [0.0] * 36
        cov[0] = 0.25
        cov[7] = 0.25
        cov[35] = 0.0685
        msg.pose.covariance = cov

        self.initial_pose_pub.publish(msg)

        self.get_logger().warn(
            f'[WARN] Published wrong initial pose: '
            f'dx={WRONG_OFFSET_X:.2f}, dy={WRONG_OFFSET_Y:.2f}, '
            f'dyaw={WRONG_OFFSET_YAW_DEG:.2f} deg'
        )

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
            'amcl_relocalization_test',
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
        if (
            self.start_amcl_pose is None or
            self.start_odom_pose is None or
            self.end_amcl_pose is None or
            self.end_odom_pose is None
        ):
            self.finish_and_exit(
                'FAIL',
                'missing pose snapshots',
                'could not record required AMCL/odom start/end poses'
            )
            return

        amcl_updates = self.end_amcl_count - self.start_amcl_count

        start_amcl_xy = (self.start_amcl_pose[0], self.start_amcl_pose[1])
        end_amcl_xy = (self.end_amcl_pose[0], self.end_amcl_pose[1])

        start_odom_xy = (self.start_odom_pose[0], self.start_odom_pose[1])
        end_odom_xy = (self.end_odom_pose[0], self.end_odom_pose[1])

        amcl_dist = euclidean_distance(start_amcl_xy, end_amcl_xy)
        odom_dist = euclidean_distance(start_odom_xy, end_odom_xy)
        dist_diff = abs(amcl_dist - odom_dist)

        amcl_yaw_change_deg = abs(normalize_angle_deg(
            math.degrees(self.end_amcl_pose[2] - self.start_amcl_pose[2])
        ))
        odom_yaw_change_deg = abs(normalize_angle_deg(
            math.degrees(self.end_odom_pose[2] - self.start_odom_pose[2])
        ))
        yaw_diff_deg = abs(amcl_yaw_change_deg - odom_yaw_change_deg)

        self.get_logger().info('========== ANALYSIS ==========')
        self.get_logger().info(f'[INFO] New AMCL messages during test: {amcl_updates}')
        self.get_logger().info(f'[INFO] AMCL distance: {amcl_dist:.3f} m')
        self.get_logger().info(f'[INFO] Odom distance: {odom_dist:.3f} m')
        self.get_logger().info(f'[INFO] Distance difference: {dist_diff:.3f} m')
        self.get_logger().info(f'[INFO] AMCL yaw change: {amcl_yaw_change_deg:.2f} deg')
        self.get_logger().info(f'[INFO] Odom yaw change: {odom_yaw_change_deg:.2f} deg')
        self.get_logger().info(f'[INFO] Yaw difference: {yaw_diff_deg:.2f} deg')

        if amcl_updates <= 0:
            self.finish_and_exit(
                'FAIL',
                'amcl did not update',
                'received no new /amcl_pose messages during relocalization test'
            )
            return

        if odom_dist < MIN_ODOM_DISTANCE_M:
            self.finish_and_exit(
                'FAIL',
                f'odom distance too small ({odom_dist:.3f} m)',
                'robot did not move enough for meaningful relocalization comparison'
            )
            return

        if dist_diff <= PASS_DIST_DIFF_M and yaw_diff_deg <= PASS_YAW_DIFF_DEG:
            status = 'PASS'
        elif dist_diff <= WARN_DIST_DIFF_M and yaw_diff_deg <= WARN_YAW_DIFF_DEG:
            status = 'WARN'
        else:
            status = 'FAIL'

        measurement = (
            f'amcl_dist={amcl_dist:.3f} m | '
            f'odom_dist={odom_dist:.3f} m | '
            f'dist_diff={dist_diff:.3f} m | '
            f'yaw_diff={yaw_diff_deg:.2f} deg'
        )

        notes = (
            f'wrong_pose_dx={WRONG_OFFSET_X:.2f}, '
            f'wrong_pose_dy={WRONG_OFFSET_Y:.2f}, '
            f'wrong_pose_dyaw={WRONG_OFFSET_YAW_DEG:.2f} deg, '
            f'amcl_updates={amcl_updates}'
        )

        self.finish_and_exit(status, measurement, notes)

    def loop(self) -> None:
        now = time.time()
        elapsed = now - self.start_time
        phase_elapsed = now - self.phase_start_time

        if self.done:
            if now - self.finish_time > 0.5:
                self.get_logger().info('Exiting amcl_relocalization_test')
                rclpy.shutdown()
            return

        if elapsed > MAX_TEST_TIME:
            self.finish_and_exit(
                'FAIL',
                'timed out',
                'relocalization test timed out'
            )
            return

        if self.phase == 'wait_for_ready':
            self.stop_robot()

            if self.have_required_topics():
                self.start_amcl_pose = self.get_amcl_pose()
                self.start_odom_pose = self.get_odom_pose()
                self.start_amcl_count = self.amcl_count

                self.get_logger().info(
                    f'[INFO] Start AMCL pose: '
                    f'x={self.start_amcl_pose[0]:.3f}, y={self.start_amcl_pose[1]:.3f}, '
                    f'yaw={math.degrees(self.start_amcl_pose[2]):.2f} deg'
                )
                self.get_logger().info(
                    f'[INFO] Start Odom pose: '
                    f'x={self.start_odom_pose[0]:.3f}, y={self.start_odom_pose[1]:.3f}, '
                    f'yaw={math.degrees(self.start_odom_pose[2]):.2f} deg'
                )

                self.transition('publish_wrong_pose')
                return

            if phase_elapsed >= STARTUP_WAIT_TIMEOUT:
                self.finish_and_exit(
                    'FAIL',
                    'startup readiness timeout',
                    f'timed out waiting for topics; amcl_msgs={self.amcl_count}, odom_msgs={self.odom_count}'
                )
                return
            return

        if self.phase == 'publish_wrong_pose':
            self.publish_wrong_initial_pose()
            self.transition('wait_after_wrong_pose')
            return

        if self.phase == 'wait_after_wrong_pose':
            self.stop_robot()
            if phase_elapsed >= POST_WRONG_POSE_SETTLE:
                self.transition('rotate')
            return

        if self.phase == 'rotate':
            if phase_elapsed < ROTATE_DURATION:
                self.publish_cmd(0.0, ROTATE_SPEED)
            else:
                self.stop_robot()
                self.transition('forward')
            return

        if self.phase == 'forward':
            if phase_elapsed < FORWARD_DURATION:
                self.publish_cmd(FORWARD_SPEED, 0.0)
            else:
                self.stop_robot()
                self.transition('settling')
            return

        if self.phase == 'settling':
            self.stop_robot()
            if phase_elapsed >= SETTLE_TIME:
                self.end_amcl_pose = self.get_amcl_pose()
                self.end_odom_pose = self.get_odom_pose()
                self.end_amcl_count = self.amcl_count

                self.get_logger().info(
                    f'[INFO] End AMCL pose: '
                    f'x={self.end_amcl_pose[0]:.3f}, y={self.end_amcl_pose[1]:.3f}, '
                    f'yaw={math.degrees(self.end_amcl_pose[2]):.2f} deg'
                )
                self.get_logger().info(
                    f'[INFO] End Odom pose: '
                    f'x={self.end_odom_pose[0]:.3f}, y={self.end_odom_pose[1]:.3f}, '
                    f'yaw={math.degrees(self.end_odom_pose[2]):.2f} deg'
                )

                self.analyze()
            return


def main(args=None):
    rclpy.init(args=args)
    node = AMCLRelocalizationTest()

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