#!/usr/bin/env python3

"""
map_odom_tf_check.py

Check TF consistency between:
- map -> odom
- odom -> base
- map -> base

Ensures the localization transform chain is available and connected.

-------------------------------------------------------------------------------
IMPORTANT NOTE ON FRAME SELECTION (THIS FIXES TF LOOKUP FAILURES)
-------------------------------------------------------------------------------

You may encounter a situation where:

- TF is clearly working (robot moves, Nav2 works, RViz looks correct)
- But this script fails with:
    [FAIL] TF lookup timeout

WHY THIS HAPPENS:

Many robots (including TurtleBot3) DO NOT use the same base frame name.

Common base frames:
    - base_footprint   (TurtleBot3)
    - base_link        (other robots)

In your system (confirmed from /odom topic):

    child_frame_id: base_footprint

So your TF chain is:
    map -> odom -> base_footprint

BUT if you hardcode:

    odom -> base_link

that transform does NOT exist → lookup fails → script incorrectly fails.

-------------------------------------------------------------------------------
THE FIX:

Instead of assuming a frame, we dynamically detect it:

    self.base_candidates = ['base_footprint', 'base_link']

Then we select whichever one actually exists at runtime.

This makes the script:
    ✔ robot-agnostic
    ✔ robust
    ✔ correct for TurtleBot3 and other robots

-------------------------------------------------------------------------------
SECONDARY FIX: SAFE TF LOOKUP

Direct TF lookups can throw exceptions if transforms are not ready yet.

We use:

    lookup_transform_safe()

which:
    - catches TF exceptions
    - allows retrying
    - prevents crashes
    - enables clean timeout behavior

-------------------------------------------------------------------------------
WHAT THIS TEST ACTUALLY VERIFIES:

This is NOT a numerical accuracy test.

It verifies:

    ✔ map -> odom exists   (provided by AMCL/localization)
    ✔ odom -> base exists  (provided by odometry/robot)
    ✔ map -> base exists   (combined TF chain)

Which confirms:

    ✔ Localization TF chain is connected and usable

-------------------------------------------------------------------------------
"""

import math
import time

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from .result_utils import append_result


class MapOdomTFCheck(Node):
    def __init__(self):
        super().__init__('map_odom_tf_check')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.start_time = time.time()
        self.timeout = 10.0

        # ===============================
        # CRITICAL FIX: DO NOT HARDCODE BASE FRAME
        # ===============================
        #
        # Different robots use different base frames.
        # TurtleBot3 uses "base_footprint", not "base_link".
        #
        # We check both and select whichever exists.
        #
        self.base_candidates = ['base_footprint', 'base_link']

        self.timer = self.create_timer(0.5, self.run)

        self.get_logger().info('========== MAP ODOM TF CHECK ==========')
        self.get_logger().info('Checking localization TF chain: map -> odom -> base')

    def lookup_transform_safe(self, parent: str, child: str):
        """
        Safe TF lookup.

        WHY:
        TF may not be immediately available at startup.
        Direct lookup can throw exceptions.

        FIX:
        - Catch TF exceptions
        - Return None instead of crashing
        - Allow retry until timeout
        """
        try:
            return self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        except Exception as exc:
            self.get_logger().warn(
                f'Unexpected TF error for {parent} -> {child}: {exc}'
            )
            return None

    def tf_summary(self, tf_msg):
        tx = tf_msg.transform.translation.x
        ty = tf_msg.transform.translation.y
        tz = tf_msg.transform.translation.z
        return f'x={tx:.3f}, y={ty:.3f}, z={tz:.3f}'

    def run(self):
        elapsed = time.time() - self.start_time

        selected_base = None
        tf_odom_base = None

        # ===============================
        # FIX: DYNAMIC BASE FRAME DETECTION
        # ===============================
        #
        # Instead of assuming "base_link", we test each candidate.
        #
        for base_frame in self.base_candidates:
            tf_candidate = self.lookup_transform_safe('odom', base_frame)
            if tf_candidate is not None:
                selected_base = base_frame
                tf_odom_base = tf_candidate
                break

        tf_map_odom = self.lookup_transform_safe('map', 'odom')
        tf_map_base = None
        if selected_base is not None:
            tf_map_base = self.lookup_transform_safe('map', selected_base)

        # ===============================
        # PASS CONDITION
        # ===============================
        if tf_map_odom is not None and tf_odom_base is not None and tf_map_base is not None:
            self.get_logger().info('[PASS] Localization TF chain resolved successfully')
            self.get_logger().info(f'[INFO] Using base frame: {selected_base}')

            self.get_logger().info('[INFO] Verified transforms:')
            self.get_logger().info(
                f'[INFO]   map -> odom: {self.tf_summary(tf_map_odom)}'
            )
            self.get_logger().info(
                f'[INFO]   odom -> {selected_base}: {self.tf_summary(tf_odom_base)}'
            )
            self.get_logger().info(
                f'[INFO]   map -> {selected_base}: {self.tf_summary(tf_map_base)}'
            )

            self.get_logger().info(
                '[INFO] This confirms the global localization chain is connected: '
                f'map -> odom -> {selected_base}'
            )

            append_result(
                'map_odom_tf_check',
                'PASS',
                'tf chain valid',
                f'verified map->odom, odom->{selected_base}, map->{selected_base}'
            )
            self.shutdown()
            return

        # ===============================
        # FAIL CONDITION (TIMEOUT)
        # ===============================
        if elapsed > self.timeout:
            missing = []

            if tf_map_odom is None:
                missing.append('map->odom')
            if tf_odom_base is None:
                missing.append('odom->base')
            if tf_map_base is None:
                missing.append('map->base')

            self.get_logger().error('[FAIL] TF lookup timeout')
            self.get_logger().error(
                '[ERROR] Could not verify full localization chain: map -> odom -> base'
            )

            if tf_map_odom is not None:
                self.get_logger().info(
                    f'[INFO] Verified map -> odom: {self.tf_summary(tf_map_odom)}'
                )

            if tf_odom_base is not None and selected_base is not None:
                self.get_logger().info(
                    f'[INFO] Verified odom -> {selected_base}: {self.tf_summary(tf_odom_base)}'
                )

            if missing:
                self.get_logger().error(
                    '[ERROR] Missing transforms: ' + ', '.join(missing)
                )

            append_result(
                'map_odom_tf_check',
                'FAIL',
                'tf chain invalid',
                'missing: ' + ', '.join(missing)
            )
            self.shutdown()

        else:
            self.get_logger().info('[INFO] Waiting for TF...')
            if tf_map_odom is None:
                self.get_logger().info('[INFO]   waiting for map -> odom')
            if tf_odom_base is None:
                self.get_logger().info('[INFO]   waiting for odom -> base')
            if selected_base is not None and tf_map_base is None:
                self.get_logger().info(f'[INFO]   waiting for map -> {selected_base}')

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomTFCheck()

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