"""
summary_report.py

Read the shared validation CSV results file and print a clean summary table.

Goal:
- Show all expected test results in one place
- Wrap long text so the table stays readable
"""

import csv
import textwrap

import rclpy
from rclpy.node import Node

from tb3_localization_validation.result_utils import RESULTS_FILE


TEST_ORDER = [

]


# Max width for each column
MAX_WIDTHS = [28, 10, 14, 60]  # Test, Status, Measurement, Notes


class SummaryReport(Node):
    def __init__(self):
        super().__init__('summary_report')
        self.print_summary()

    def print_summary(self):
        """
        Read the CSV results file and print a formatted terminal table.
        """
        results = {}

        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    self.get_logger().warn('Results file is empty or missing a header row.')
                elif 'test' not in reader.fieldnames:
                    self.get_logger().warn(
                        f'Invalid CSV header in {RESULTS_FILE}. Found: {reader.fieldnames}'
                    )
                else:
                    for row in reader:
                        test_name = row.get('test', '').strip()
                        if test_name:
                            results[test_name] = row
        else:
            self.get_logger().warn(f'Results file not found: {RESULTS_FILE}')

        rows = []
        for test_name in TEST_ORDER:
            if test_name in results:
                row = results[test_name]
                rows.append([
                    row.get('test', test_name),
                    row.get('status', 'UNKNOWN'),
                    row.get('measurement', ''),
                    row.get('notes', '')
                ])
            else:
                rows.append([test_name, 'MISSING', '', 'no result found'])

        headers = ['Test', 'Status', 'Measurement', 'Notes']

        def wrap_cell(text, width):
            return textwrap.wrap(str(text), width=width) or ['']

        # Wrap headers
        wrapped_headers = [
            wrap_cell(headers[i], MAX_WIDTHS[i]) for i in range(len(headers))
        ]
        max_header_lines = max(len(cell) for cell in wrapped_headers)
        for cell in wrapped_headers:
            while len(cell) < max_header_lines:
                cell.append('')

        # Wrap rows
        wrapped_rows = []
        for row in rows:
            wrapped = [
                wrap_cell(row[i], MAX_WIDTHS[i]) for i in range(len(row))
            ]
            max_lines = max(len(cell) for cell in wrapped)
            for cell in wrapped:
                while len(cell) < max_lines:
                    cell.append('')
            wrapped_rows.append(wrapped)

        def format_wrapped_row(wrapped_row):
            lines = []
            num_lines = len(wrapped_row[0])

            for line_idx in range(num_lines):
                line = '| ' + ' | '.join(
                    wrapped_row[col_idx][line_idx].ljust(MAX_WIDTHS[col_idx])
                    for col_idx in range(len(wrapped_row))
                ) + ' |'
                lines.append(line)

            return '\n'.join(lines)

        border = '+-' + '-+-'.join('-' * w for w in MAX_WIDTHS) + '-+'

        print()
        print('========================================')
        print('TurtleBot3 Sensors Validation Summary')
        print('========================================')
        print(border)
        print(format_wrapped_row(wrapped_headers))
        print(border)

        for wrapped_row in wrapped_rows:
            print(format_wrapped_row(wrapped_row))
            print(border)

        print()


def main(args=None):
    rclpy.init(args=args)
    node = SummaryReport()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()