#!/usr/bin/env python3

"""
amcl_translation_tracking.py

Command a forward motion and compare translation tracking between:
- AMCL (/amcl_pose)
- Odom (/odom)

Goal:
- Verify AMCL updates during forward motion
- Check whether AMCL translation roughly agrees with odom translation

-------------------------------------------------------------------------------
IMPORTANT NOTE ON THE FIRST FIX: /amcl_pose QoS
-------------------------------------------------------------------------------

A common ROS 2 failure mode is:

    ros2 topic echo /amcl_pose --once

works fine, but this node appears to receive no AMCL data.

This usually does NOT mean AMCL is broken.

WHY THIS HAPPENS:
- /amcl_pose often behaves like a latched-style topic
- In ROS 2 terms, that typically means TRANSIENT_LOCAL durability
- A normal VOLATILE subscriber may miss the stored AMCL pose
- As a result, the script may wait forever or only receive too few AMCL updates

THE FIX:
We explicitly subscribe to /amcl_pose using:

    reliability = RELIABLE
    durability = TRANSIENT_LOCAL

This tells ROS 2:
    "Give me the latest stored /amcl_pose message immediately,
     even if it was published before this node started."

For /odom:
- use sensor-data QoS because it is a high-rate streaming topic

-------------------------------------------------------------------------------
IMPORTANT NOTE ON THE SECOND FIX: MOTION SIZE / AMCL UPDATE THRESHOLD
-------------------------------------------------------------------------------

A second issue found during testing was:

- The robot visibly moved forward
- /odom updated normally
- But AMCL did not publish a new pose during the motion window
- The script reported:
      New AMCL messages during test: 0

This usually does NOT mean the AMCL subscriber is broken.

AMCL often updates only after motion exceeds internal thresholds such as:

    update_min_d   (minimum translation before AMCL updates)
    update_min_a   (minimum rotation before AMCL updates)

So if the robot moves too little:
- odom may clearly change
- but AMCL may still publish no new pose

That is exactly what happened in earlier testing:
- 0.08 m/s for 3.0 sec -> about 0.24 m
- odom changed
- AMCL did not update

THE FIX:
Increase forward motion so it clearly exceeds the AMCL translation threshold.

Working example from testing:
- 0.08 m/s for 5.0 sec -> about 0.40 m
- this produced a new AMCL update

This is why the script now uses a longer forward duration.

-------------------------------------------------------------------------------
IMPORTANT NOTE ON SIMULATION
-------------------------------------------------------------------------------

For simulation-based localization testing, Nav2/AMCL should be launched with a
valid map YAML. Otherwise /amcl_pose may not behave correctly.

Example:
    ros2 launch turtlebot3_navigation2 navigation2.launch.py \\
        map:=$HOME/map_turtlebot3_world.yaml use_sim_time:=true

-------------------------------------------------------------------------------
WHAT THIS TEST MEASURES
-------------------------------------------------------------------------------

This is not a perfect localization accuracy test.

It measures whether:
- AMCL position changes when the robot moves forward
- Odom position changes when the robot moves forward
- The AMCL translation is reasonably close to odom translation

This is useful before Nav2 because it tells you whether localization is
responsive during translation.
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
CMD_VEL_TOPIC = '/cmd_vel'

# ===== Timing =====
CONTROL_PERIOD = 0.05
PROGRESS_PERIOD = 1.0
STARTUP_WAIT_TIMEOUT = 15.0
MAX_TEST_TIME = 40.0
SETTLE_TIME = 3.0

# ===== Motion =====
FORWARD_SPEED = 0.08       # m/s

# CRITICAL FIX:
# Earlier testing showed that ~0.24 m of forward motion was sometimes not enough
# to trigger a new AMCL update. Odom changed, but AMCL message count stayed flat.
#
# AMCL often uses a minimum translation threshold (update_min_d), so the robot
# must move far enough before AMCL publishes an updated pose.
#
# Using 5.0 sec at 0.08 m/s gives about 0.40 m of travel, which was enough to
# produce a new AMCL update in testing.
FORWARD_DURATION = 5.0     # sec

# ===== Thresholds =====
MIN_ODOM_DISTANCE_M = 0.20
PASS_DIFF_M = 0.10
WARN_DIFF_M = 0.20


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


class AMCLTranslationTracking(Node):
    def __init__(self) -> None:
        super().__init__('amcl_translation_tracking')

        # =====================================================================
        # CRITICAL FIX: QoS for /amcl_pose
        # =====================================================================
        #
        # /amcl_pose may behave like a transient-local (latched-style) topic.
        # If we subscribe with default QoS, this node may receive too few AMCL
        # updates or only the stored initial pose.
        #
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

        self.start_amcl_xy: Optional[Tuple[float, float]] = None
        self.start_odom_xy: Optional[Tuple[float, float]] = None
        self.end_amcl_xy: Optional[Tuple[float, float]] = None
        self.end_odom_xy: Optional[Tuple[float, float]] = None

        # Track AMCL message count across the motion window
        self.start_amcl_count = 0
        self.end_amcl_count = 0

        self.timer = self.create_timer(CONTROL_PERIOD, self.loop)
        self.progress_timer = self.create_timer(PROGRESS_PERIOD, self.progress_update)

        self.get_logger().info('========== AMCL TRANSLATION TRACKING ==========')
        self.get_logger().info(f'AMCL topic: {AMCL_TOPIC}')
        self.get_logger().info(f'Odom topic: {ODOM_TOPIC}')
        self.get_logger().info(f'Command topic: {CMD_VEL_TOPIC}')
        self.get_logger().info(f'Forward speed: {FORWARD_SPEED:.2f} m/s')
        self.get_logger().info(f'Forward duration: {FORWARD_DURATION:.2f} sec')
        self.get_logger().info(f'Expected motion: {FORWARD_SPEED * FORWARD_DURATION:.2f} m')
        self.get_logger().info(f'Settle time: {SETTLE_TIME:.2f} sec')

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

    def get_amcl_xy(self) -> Optional[Tuple[float, float]]:
        if self.amcl_msg is None:
            return None
        p = self.amcl_msg.pose.pose.position
        return (p.x, p.y)

    def get_odom_xy(self) -> Optional[Tuple[float, float]]:
        if self.odom_msg is None:
            return None
        p = self.odom_msg.pose.pose.position
        return (p.x, p.y)

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
            'amcl_translation_tracking',
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
            self.start_amcl_xy is None or
            self.start_odom_xy is None or
            self.end_amcl_xy is None or
            self.end_odom_xy is None
        ):
            self.finish_and_exit(
                'FAIL',
                'missing position snapshots',
                'could not record required AMCL/odom position values'
            )
            return

        # If AMCL did not publish any new poses during motion, the most likely
        # cause is that commanded travel stayed below AMCL's motion update
        # threshold rather than a pure subscriber failure.
        amcl_updates_during_test = self.end_amcl_count - self.start_amcl_count

        self.get_logger().info(f'[INFO] AMCL messages at start: {self.start_amcl_count}')
        self.get_logger().info(f'[INFO] AMCL messages at end: {self.end_amcl_count}')
        self.get_logger().info(f'[INFO] New AMCL messages during test: {amcl_updates_during_test}')

        if amcl_updates_during_test <= 0:
            self.finish_and_exit(
                'FAIL',
                'amcl did not update',
                'received no new /amcl_pose messages; likely motion stayed below AMCL update_min_d'
            )
            return

        amcl_dist = euclidean_distance(self.start_amcl_xy, self.end_amcl_xy)
        odom_dist = euclidean_distance(self.start_odom_xy, self.end_odom_xy)
        diff_m = abs(amcl_dist - odom_dist)

        self.get_logger().info(f'[INFO] AMCL distance: {amcl_dist:.3f} m')
        self.get_logger().info(f'[INFO] Odom distance: {odom_dist:.3f} m')
        self.get_logger().info(f'[INFO] Absolute difference: {diff_m:.3f} m')

        if odom_dist < MIN_ODOM_DISTANCE_M:
            self.finish_and_exit(
                'FAIL',
                f'odom distance too small ({odom_dist:.3f} m)',
                'robot did not move enough for meaningful comparison'
            )
            return

        if diff_m <= PASS_DIFF_M:
            status = 'PASS'
        elif diff_m <= WARN_DIFF_M:
            status = 'WARN'
        else:
            status = 'FAIL'

        measurement = (
            f'amcl={amcl_dist:.3f} m | '
            f'odom={odom_dist:.3f} m | '
            f'diff={diff_m:.3f} m'
        )
        notes = (
            f'forward_speed={FORWARD_SPEED:.2f} m/s, '
            f'forward_duration={FORWARD_DURATION:.2f} sec, '
            f'amcl_updates={amcl_updates_during_test}'
        )

        if status == 'PASS':
            self.get_logger().info('[PASS] AMCL translation tracking looks good')
        elif status == 'WARN':
            self.get_logger().warn('[WARN] AMCL translation tracking is usable but not ideal')
        else:
            self.get_logger().error('[FAIL] AMCL translation tracking differs too much from odom')

        self.finish_and_exit(status, measurement, notes)

    def loop(self) -> None:
        now = time.time()
        elapsed = now - self.start_time
        phase_elapsed = now - self.phase_start_time

        if self.done:
            if now - self.finish_time > 0.5:
                self.get_logger().info('Exiting amcl_translation_tracking')
                rclpy.shutdown()
            return

        if elapsed > MAX_TEST_TIME:
            self.finish_and_exit(
                'FAIL',
                'timed out',
                'translation tracking test timed out'
            )
            return

        if self.phase == 'wait_for_ready':
            self.stop_robot()

            if self.have_required_topics():
                self.start_amcl_xy = self.get_amcl_xy()
                self.start_odom_xy = self.get_odom_xy()
                self.start_amcl_count = self.amcl_count

                if self.start_amcl_xy is None or self.start_odom_xy is None:
                    return

                self.get_logger().info(
                    f'[INFO] Start AMCL position: x={self.start_amcl_xy[0]:.3f}, y={self.start_amcl_xy[1]:.3f}'
                )
                self.get_logger().info(
                    f'[INFO] Start Odom position: x={self.start_odom_xy[0]:.3f}, y={self.start_odom_xy[1]:.3f}'
                )
                self.get_logger().info(f'[INFO] Start AMCL message count: {self.start_amcl_count}')
                self.transition('moving_forward')
                return

            if phase_elapsed >= STARTUP_WAIT_TIMEOUT:
                self.finish_and_exit(
                    'FAIL',
                    'startup readiness timeout',
                    f'timed out waiting for topics; amcl_msgs={self.amcl_count}, odom_msgs={self.odom_count}'
                )
                return
            return

        if self.phase == 'moving_forward':
            if phase_elapsed < FORWARD_DURATION:
                self.publish_cmd(FORWARD_SPEED, 0.0)
            else:
                self.stop_robot()
                self.transition('settling')
            return

        if self.phase == 'settling':
            self.stop_robot()
            if phase_elapsed >= SETTLE_TIME:
                self.end_amcl_xy = self.get_amcl_xy()
                self.end_odom_xy = self.get_odom_xy()
                self.end_amcl_count = self.amcl_count

                self.get_logger().info(
                    f'[INFO] End AMCL position: x={self.end_amcl_xy[0]:.3f}, y={self.end_amcl_xy[1]:.3f}'
                )
                self.get_logger().info(
                    f'[INFO] End Odom position: x={self.end_odom_xy[0]:.3f}, y={self.end_odom_xy[1]:.3f}'
                )
                self.get_logger().info(f'[INFO] End AMCL message count: {self.end_amcl_count}')

                self.analyze()
            return


def main(args=None):
    rclpy.init(args=args)
    node = AMCLTranslationTracking()

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