import math
import statistics
from collections import Counter


def describe(records):
    return {'failure_records': sum(row['failed'] for row in records), 'devices': len({row['device'] for row in records}), 'temperature_max': max(row['temperature'] for row in records)}
