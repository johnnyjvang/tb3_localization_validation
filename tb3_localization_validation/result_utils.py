"""
result_utils.py

Utility functions for managing the validation results CSV file.

Goal:
- Create/reset the results file
- Append standardized test rows
- Keep all package test outputs consistent
"""

import csv
from pathlib import Path


# Store results in /tmp so each test script can easily share one file
RESULTS_DIR = Path('/tmp/tb3_localization_validation')
RESULTS_FILE = RESULTS_DIR / 'results.csv'


def reset_results_file():
    """
    Reset the results CSV file and write the header row.

    This should be called before running a full validation launch sequence.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test', 'status', 'measurement', 'notes'])


def append_result(test_name, status, measurement, notes=''):
    """
    Append one standardized test result row to the CSV file.

    Parameters
    ----------
    test_name : str
        Name of the test, such as 'imu_message_rate'
    status : str
        PASS / FAIL / WARN / MISSING
    measurement : str
        Main result value, such as '198.52 Hz'
    notes : str
        Optional additional detail
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([test_name, status, measurement, notes])