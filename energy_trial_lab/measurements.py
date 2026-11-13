import math
from collections import defaultdict
import numpy as np


def net_energy(row, baselines):
    baseline = baselines[row['batch_id']]
    numbers = [row['power_mw'], row['duration_ms'], baseline['power_mw'], baseline['se_mw']]
    if any(not math.isfinite(value) or value < 0 for value in numbers):
        raise ValueError('测量值无效')
    return (row['power_mw'] - baseline['power_mw']) * row['duration_ms'] / 1000


def trial_keys(rows):
    keys = [(row['device_id'], row['task'], row['batch_id'], row['config'], row['repeat']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('实验记录重复')
    return keys


def inspect(document):
    rows = document['trials']
    trial_keys(rows)
    groups = defaultdict(list)
    baselines = document['baselines']
    for row in rows:
        groups[row['config']].append((row, net_energy(row, baselines)))
    result = {}
    for config, values in sorted(groups.items()):
        energies = [energy for row, energy in values]
        completed = [row['duration_ms'] for row, energy in values if row['completed']]
        result[config] = {'runs': len(values), 'completion_rate': float(np.mean([row['completed'] for row, energy in values])),
                          'observed_mean_energy_mj': float(np.mean(energies)),
                          'completed_median_ms': float(np.median(completed)) if completed else None}
    return {'observed': result, 'measurement_batches': sorted(baselines)}
