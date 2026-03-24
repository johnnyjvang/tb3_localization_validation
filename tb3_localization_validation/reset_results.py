"""
reset_results.py

Reset the tb3_localization_validation results file before running a test suite.

Goal:
- Clear out old CSV results
- Start with a fresh file for the current validation run
"""

import rclpy
from rclpy.node import Node

from tb3_localization_validation.result_utils import reset_results_file


class ResetResults(Node):
    def __init__(self):
        super().__init__('reset_results')

        reset_results_file()
        self.get_logger().info('Reset results file for tb3_localization_validation')


def main(args=None):
    rclpy.init(args=args)
    node = ResetResults()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()