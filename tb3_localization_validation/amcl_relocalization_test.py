#!/usr/bin/env python3

"""
amcl_relocalization_test.py

Test whether AMCL can recover after being intentionally given a wrong initial pose.

UPDATED FIXES:
1. Add a small motion sequence after publishing the wrong pose, because
   stationary relocalization often does not recover well.
2. Throttle recovery logging so the node does not spam the terminal.
3. Make shutdown safe so Ctrl+C does not cause a publisher context error.

Why the RViz map jumps:
- Publishing a wrong /initialpose causes AMCL to change the map->odom transform.
- In RViz, that often looks like the map shifts or rotates.
- That is expected for this test.

Why the old version spammed:
- The recovery loop ran at 20 Hz and printed on every cycle.
- This version prints recovery status at most once per second.

Why the old version crashed on Ctrl+C:
- The node tried to publish a stop command after ROS had already begun shutdown.
- This version checks whether the ROS context is still valid before publishing.
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
INITIALPOSE_TOPIC = '/initialpose'
ODOM_TOPIC = '/odom'
CMD_VEL_TOPIC = '/cmd_vel'

# ===== Timing =====
CONTROL_PERIOD = 0.05
PROGRESS_PERIOD = 1.0
STARTUP_WAIT_TIMEOUT = 15.0
POST_WRONG_POSE_SETTLE = 2.0
RECOVERY_TIMEOUT = 25.0
MAX_TEST_TIME = 60.0
SETTLE_BETWEEN_MOTIONS = 1.0

# ===== Wrong initial pose offset =====
WRONG_OFFSET_X = 1.0
WRONG_OFFSET_Y = 0.5
WRONG_OFFSET_YAW_DEG = 30.0

# ===== Recovery thresholds =====
PASS_POS_ERROR_M = 0.20
WARN_POS_ERROR_M = 0.50
PASS_YAW_ERROR_DEG = 10.0
WARN_YAW_ERROR_DEG = 25.0

# ===== Motion sequence to help relocalization =====
ROTATE_SPEED = 0.35
ROTATE_DURATION_1 = 2.0
FORWARD_SPEED = 0.06
FORWARD_DURATION = 3.0
ROTATE_DURATION_2 = 2.0


def yaw_to_quaternion(yaw_rad: float) -> Tuple[float, float]:
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return qz, qw


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


class AMCLRelocalizationTest(Node):
    def __init__(self) -> None:
        super().__init__('amcl_relocalization_test')

        amcl_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.latest_amcl_pose: Optional[PoseWithCovarianceStamped] = None
        self.latest_odom: Optional[Odometry] = None
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

        self.reference_pose: Optional[Tuple[float, float, float]] = None
        self.recovery_start_time = None
        self.recovered_time = None

        self.start_amcl_count = 0
        self.end_amcl_count = 0
        self.last_recovery_log_time = 0.0

        self.timer = self.create_timer(CONTROL_PERIOD, self.loop)
        self.progress_timer = self.create_timer(PROGRESS_PERIOD, self.progress_update)

        self.get_logger().info('========== AMCL RELOCALIZATION TEST ==========')
        self.get_logger().info(f'AMCL topic: {AMCL_TOPIC}')
        self.get_logger().info(f'Initial pose topic: {INITIALPOSE_TOPIC}')
        self.get_logger().info(f'Odom topic: {ODOM_TOPIC}')
        self.get_logger().info(f'Command topic: {CMD_VEL_TOPIC}')

    def amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self.latest_amcl_pose = msg
        self.amcl_count += 1

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.odom_count += 1

    def get_amcl_pose(self) -> Optional[Tuple[float, float, float]]:
        if self.latest_amcl_pose is None:
            return None
        p = self.latest_amcl_pose.pose.pose.position
        q = self.latest_amcl_pose.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        return (p.x, p.y, yaw)

    def build_wrong_initial_pose(self, ref_x: float, ref_y: float, ref_yaw_deg: float):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = ref_x + WRONG_OFFSET_X
        msg.pose.pose.position.y = ref_y + WRONG_OFFSET_Y
        msg.pose.pose.position.z = 0.0

        yaw_deg = ref_yaw_deg + WRONG_OFFSET_YAW_DEG
        yaw_rad = math.radians(yaw_deg)
        qz, qw = yaw_to_quaternion(yaw_rad)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        cov = [0.0] * 36
        cov[0] = 0.25
        cov[7] = 0.25
        cov[35] = 0.0685
        msg.pose.covariance = cov
        return msg

    def publish_wrong_initial_pose(self) -> None:
        ref_x, ref_y, ref_yaw = self.reference_pose
        ref_yaw_deg = math.degrees(ref_yaw)

        msg = self.build_wrong_initial_pose(ref_x, ref_y, ref_yaw_deg)
        self.initial_pose_pub.publish(msg)

        self.start_amcl_count = self.amcl_count

        self.get_logger().warn(
            '[WARN] Published intentionally wrong initial pose '
            f'(dx={WRONG_OFFSET_X:.2f}, dy={WRONG_OFFSET_Y:.2f}, '
            f'dyaw={WRONG_OFFSET_YAW_DEG:.2f} deg)'
        )

    def publish_cmd(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        if not rclpy.ok():
            return
        msg = TwistStamped()
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        try:
            self.cmd_pub.publish(msg)
        except Exception:
            pass

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)

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

    def analyze_current_error(self) -> Optional[Tuple[float, float]]:
        current = self.get_amcl_pose()
        if current is None or self.reference_pose is None:
            return None

        cx, cy, cyaw = current
        rx, ry, ryaw = self.reference_pose

        pos_error = euclidean_distance(rx, ry, cx, cy)
        yaw_error = abs(normalize_angle_deg(math.degrees(cyaw - ryaw)))
        return pos_error, yaw_error

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
            pose = self.get_amcl_pose()
            if pose is not None:
                self.reference_pose = pose
                x, y, yaw = pose
                self.get_logger().info(
                    f'[INFO] Reference AMCL pose: x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.2f} deg'
                )
                self.transition('publish_wrong_pose')
                return

            if phase_elapsed >= STARTUP_WAIT_TIMEOUT:
                self.finish_and_exit(
                    'FAIL',
                    'amcl not initialized',
                    'did not receive /amcl_pose; make sure Nav2 is launched with a valid map and initial pose has been set'
                )
                return
            return

        if self.phase == 'publish_wrong_pose':
            self.publish_wrong_initial_pose()
            self.recovery_start_time = time.time()
            self.last_recovery_log_time = 0.0
            self.transition('wait_after_wrong_pose')
            return

        if self.phase == 'wait_after_wrong_pose':
            if phase_elapsed >= POST_WRONG_POSE_SETTLE:
                self.transition('rotate_1')
            return

        if self.phase == 'rotate_1':
            if phase_elapsed < ROTATE_DURATION_1:
                self.publish_cmd(0.0, ROTATE_SPEED)
            else:
                self.stop_robot()
                self.transition('settle_after_rotate_1')
            return

        if self.phase == 'settle_after_rotate_1':
            self.stop_robot()
            if phase_elapsed >= SETTLE_BETWEEN_MOTIONS:
                self.transition('forward')
            return

        if self.phase == 'forward':
            if phase_elapsed < FORWARD_DURATION:
                self.publish_cmd(FORWARD_SPEED, 0.0)
            else:
                self.stop_robot()
                self.transition('settle_after_forward')
            return

        if self.phase == 'settle_after_forward':
            self.stop_robot()
            if phase_elapsed >= SETTLE_BETWEEN_MOTIONS:
                self.transition('rotate_2')
            return

        if self.phase == 'rotate_2':
            if phase_elapsed < ROTATE_DURATION_2:
                self.publish_cmd(0.0, -ROTATE_SPEED)
            else:
                self.stop_robot()
                self.transition('check_recovery')
            return

        if self.phase == 'check_recovery':
            errors = self.analyze_current_error()
            if errors is None:
                return

            pos_error, yaw_error = errors
            self.end_amcl_count = self.amcl_count
            amcl_updates = self.end_amcl_count - self.start_amcl_count

            # Throttle recovery logs to once per second
            if now - self.last_recovery_log_time >= 1.0:
                self.last_recovery_log_time = now
                self.get_logger().info(
                    f'[INFO] Current recovery error: pos={pos_error:.3f} m, yaw={yaw_error:.2f} deg'
                )
                self.get_logger().info(
                    f'[INFO] New AMCL messages since wrong pose: {amcl_updates}'
                )

            if pos_error <= PASS_POS_ERROR_M and yaw_error <= PASS_YAW_ERROR_DEG:
                self.recovered_time = time.time()
                recovery_time = self.recovered_time - self.recovery_start_time
                self.finish_and_exit(
                    'PASS',
                    f'recovery_time={recovery_time:.2f} s',
                    f'final_pos_error={pos_error:.3f} m, final_yaw_error={yaw_error:.2f} deg, amcl_updates={amcl_updates}'
                )
                return

            if phase_elapsed >= RECOVERY_TIMEOUT:
                if pos_error <= WARN_POS_ERROR_M and yaw_error <= WARN_YAW_ERROR_DEG:
                    status = 'WARN'
                else:
                    status = 'FAIL'

                self.finish_and_exit(
                    status,
                    f'no full recovery in {RECOVERY_TIMEOUT:.1f} s',
                    f'final_pos_error={pos_error:.3f} m, final_yaw_error={yaw_error:.2f} deg, amcl_updates={amcl_updates}'
                )
                return


def main(args=None):
    rclpy.init(args=args)
    node = AMCLRelocalizationTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.stop_robot()
            except Exception:
                pass
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()