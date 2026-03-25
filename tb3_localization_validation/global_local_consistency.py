#!/usr/bin/env python3

"""
global_local_consistency.py

Check consistency between:
- map -> odom
- odom -> base_footprint (or base_link)
- map -> base_footprint (or base_link)

This is a localization sanity check to confirm the TF chain is resolvable
and updating as expected.
"""

import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class GlobalLocalConsistency(Node):
    def __init__(self) -> None:
        super().__init__('global_local_consistency')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_candidates = ['base_footprint', 'base_link']
        self.selected_base: Optional[str] = None

        self.start_time = time.time()
        self.timeout_sec = 15.0
        self.timer = self.create_timer(0.5, self.run)

        self.get_logger().info('========== GLOBAL LOCAL CONSISTENCY ==========')

    def get_transform(self, parent: str, child: str) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        except Exception as exc:
            self.get_logger().warn(
                f'Unexpected error getting transform {parent} -> {child}: {exc}'
            )
            return None

    def quaternion_to_yaw(self, z: float, w: float) -> float:
        return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)

    def choose_base_frame(self) -> Optional[str]:
        for frame in self.base_candidates:
            if self.get_transform('odom', frame) is not None:
                return frame
        return None

    def run(self) -> None:
        elapsed = time.time() - self.start_time

        if elapsed > self.timeout_sec:
            self.get_logger().error('[FAIL] Timed out waiting for required transforms')
            self.shutdown()
            return

        if self.selected_base is None:
            self.selected_base = self.choose_base_frame()
            if self.selected_base is None:
                self.get_logger().info('[INFO] Waiting for odom -> base frame...')
                return
            self.get_logger().info(f'[INFO] Using base frame: {self.selected_base}')

        tf_map_odom = self.get_transform('map', 'odom')
        tf_odom_base = self.get_transform('odom', self.selected_base)
        tf_map_base = self.get_transform('map', self.selected_base)

        if tf_map_odom is None:
            self.get_logger().info('[INFO] Waiting for map -> odom...')
            return

        if tf_odom_base is None:
            self.get_logger().info(f'[INFO] Waiting for odom -> {self.selected_base}...')
            return

        if tf_map_base is None:
            self.get_logger().info(f'[INFO] Waiting for map -> {self.selected_base}...')
            return

        map_base_x = tf_map_base.transform.translation.x
        map_base_y = tf_map_base.transform.translation.y
        map_base_yaw = math.degrees(
            self.quaternion_to_yaw(
                tf_map_base.transform.rotation.z,
                tf_map_base.transform.rotation.w
            )
        )

        self.get_logger().info('[PASS] All required localization transforms resolved')
        self.get_logger().info(f'[INFO] map -> odom available')
        self.get_logger().info(f'[INFO] odom -> {self.selected_base} available')
        self.get_logger().info(f'[INFO] map -> {self.selected_base} available')
        self.get_logger().info(
            f'[INFO] Current global pose: x={map_base_x:.3f}, y={map_base_y:.3f}, yaw={map_base_yaw:.2f} deg'
        )

        self.shutdown()

    def shutdown(self) -> None:
        self.get_logger().info('Global/local consistency check complete. Shutting down.')
        self.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalLocalConsistency()

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