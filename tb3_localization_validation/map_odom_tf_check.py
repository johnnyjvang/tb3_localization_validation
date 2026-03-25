#!/usr/bin/env python3

"""
map_odom_tf_check.py

Check TF consistency between:
- map -> odom
- odom -> base
- map -> base

Ensures localization transform chain is valid.
"""

import time
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from .result_utils import write_result


class MapOdomTFCheck(Node):
    def __init__(self):
        super().__init__('map_odom_tf_check')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.start_time = time.time()
        self.timeout = 10.0

        self.timer = self.create_timer(0.5, self.run)

        self.get_logger().info('========== MAP ODOM TF CHECK ==========')

    def run(self):
        elapsed = time.time() - self.start_time

        try:
            tf1 = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
            tf2 = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            tf3 = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())

            self.get_logger().info('[PASS] All TF transforms resolved')

            write_result('map_odom_tf_check', 'PASS', 1.0, 'tf valid')
            self.shutdown()

        except Exception:
            if elapsed > self.timeout:
                self.get_logger().error('[FAIL] TF lookup timeout')
                write_result('map_odom_tf_check', 'FAIL', 0.0, 'missing tf')
                self.shutdown()
            else:
                self.get_logger().info('[INFO] Waiting for TF...')

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomTFCheck()
    rclpy.spin(node)


if __name__ == '__main__':
    main()