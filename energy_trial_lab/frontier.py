"""Multi-objective selection analysis for archived terminal trials.

The analysis keeps three task-level outcomes together:

* net energy per attempt, in mJ, using the archived baseline correction;
* completion latency as a right-censored restricted mean survival time (RMST);
* completion probability.

Configuration comparisons use only blocks (device, task, batch) in which every
analysed configuration was measured.  Workloads only reweight those fixed
task-level results; they never change the support or select a favourable
subset of runs.
"""

import itertools
import math
import sys
from collections import defaultdict

import numpy as np

from .io import read_json, write_json
from .measurements import net_energy, trial_keys

EPS = 1e-9
DEFAULT_REPLICATES = 4000
CORRECTION_Z = 1.96


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), name + ' 必须是数值')
    _require(math.isfinite(value), name + ' 必须是有限数值')
    return float(value)


def _validate(document):
    _require(isinstance(document, dict), '输入必须是 JSON 对象')
    rows = document.get('trials')
    baselines = document.get('baselines')
    _require(isinstance(rows, list) and rows, 'trials 必须是非空数组')
    _require(isinstance(baselines, dict) and baselines, 'baselines 必须是非空对象')
    trial_keys(rows)

    for batch, baseline in baselines.items():
        _require(isinstance(baseline, dict), 'baseline 必须是对象：' + str(batch))
        _finite(baseline.get('power_mw'), 'baseline.power_mw')
        se = _finite(baseline.get('se_mw'), 'baseline.se_mw')
        _require(se >= 0, 'baseline.se_mw 不能为负')

    for row in rows:
        _require(isinstance(row, dict), 'trial 必须是对象')
        for field in ('device_id', 'task', 'config', 'batch_id'):
            _require(isinstance(row.get(field), str) and row[field], '缺少字符串字段：' + field)
        _require(isinstance(row.get('repeat'), int) and not isinstance(row['repeat'], bool),
                 'repeat 必须是整数')
        _require(row['batch_id'] in baselines, '未知实验批次：' + row['batch_id'])
        _require(isinstance(row.get('completed'), bool), 'completed 必须是布尔值')
        _finite(row.get('power_mw'), 'power_mw')
        _finite(row.get('duration_ms'), 'duration_ms')
        _finite(row.get('deadline_ms'), 'deadline_ms')
        net_energy(row, baselines)

    workloads = document.get('workloads')
    _require(isinstance(workloads, list) and workloads, 'workloads 必须是非空数组')
    seen = set()
    known_tasks = {row['task'] for row in rows}
    for workload in workloads:
        _require(isinstance(workload, dict), 'workload 必须是对象')
        workload_id = workload.get('id')
        _require(isinstance(workload_id, str) and workload_id, 'workload.id 必须是非空字符串')
        _require(workload_id not in seen, 'workload.id 重复：' + workload_id)
        seen.add(workload_id)
        weights = workload.get('weights')
        _require(isinstance(weights, dict) and weights, 'workload.weights 必须是非空对象')
        total = 0.0
        for task, weight in weights.items():
            _require(task in known_tasks, '工作负载引用了不存在的任务：' + str(task))
            value = _finite(weight, 'workload.weight')
            _require(value >= 0, '工作负载权重不能为负')
            total += value
        _require(total > 0, '工作负载权重之和必须大于零')

    minimum = _finite(document.get('minimum_completion_rate'), 'minimum_completion_rate')
    _require(0 <= minimum <= 1, 'minimum_completion_rate 必须位于 [0, 1]')
    seed = document.get('seed', 0)
    _require(isinstance(seed, int) and not isinstance(seed, bool), 'seed 必须是整数')
    return rows, baselines, workloads, minimum, seed


def _weights(workload):
    weights = {task: float(value) for task, value in workload['weights'].items() if value > 0}
    total = sum(weights.values())
    return {task: value / total for task, value in weights.items()}, total


def _block_key(row):
    return (row['device_id'], row['task'], row['batch_id'])


def _group_blocks(rows):
    blocks = defaultdict(lambda: defaultdict(list))
    for row in rows:
        blocks[_block_key(row)][row['config']].append(row)
    for key in blocks:
        for config in blocks[key]:
            blocks[key][config].sort(key=lambda row: row['repeat'])
    return blocks


def _rmst(durations, completed, horizon):
    """Kaplan-Meier restricted mean survival time to a fixed horizon."""
    pairs = sorted(zip(durations, completed))
    if not pairs or horizon <= 0:
        return 0.0
    at_risk = len(pairs)
    survival = 1.0
    previous = 0.0
    area = 0.0
    index = 0
    while index < len(pairs):
        time = pairs[index][0]
        if time > horizon:
            break
        area += survival * (time - previous)
        previous = time
        end = index
        events = 0
        while end < len(pairs) and pairs[end][0] == time:
            if pairs[end][1]:
                events += 1
            end += 1
        if at_risk > 0 and events:
            survival *= 1.0 - events / at_risk
        at_risk -= end - index
        index = end
    if horizon > previous:
        area += survival * (horizon - previous)
    return max(0.0, area)


def _energy(row, baselines, delta=0.0):
    if delta == 0.0:
        return net_energy(row, baselines)
    batch = row['batch_id']
    return (row['power_mw'] - baselines[batch]['power_mw'] - delta) * row['duration_ms'] / 1000.0


def _censoring_trace(rows):
    """Every uncompleted attempt, kept in the denominator of all three metrics."""
    records = []
    for row in sorted(rows, key=lambda item: (item['device_id'], item['task'], item['batch_id'],
                                              item['config'], item['repeat'])):
        if row['completed']:
            continue
        records.append({
            'device_id': row['device_id'],
            'task': row['task'],
            'batch_id': row['batch_id'],
            'config': row['config'],
            'repeat': row['repeat'],
            'stop_at_ms': row['duration_ms'],
            'deadline_ms': row['deadline_ms'],
            'reached_deadline': row['duration_ms'] >= row['deadline_ms'],
            'counted_in': ['completion_probability', 'net_energy_mj_per_attempt', 'latency_rmst_ms'],
            'latency_treatment': 'right-censored at stop_at_ms; duration is not a completion latency',
            'stop_reason': 'unknown; the archive does not distinguish cancellation from timeout',
        })
    return records


def _observed_metrics(rows, baselines, horizon, delta=None):
    delta = delta or {}
    if not rows:
        return None
    energies = [_energy(row, baselines, delta.get(row['batch_id'], 0.0)) for row in rows]
    completed = [row['completed'] for row in rows]
    completed_durations = [row['duration_ms'] for row in rows if row['completed']]
    return {
        'attempts': len(rows),
        'completed': int(sum(completed)),
        'cancelled_or_uncompleted': int(len(rows) - sum(completed)),
        'completion_probability': float(np.mean(completed)),
        'net_energy_mj_per_attempt': float(np.mean(energies)),
        'latency_rmst_ms': float(_rmst([row['duration_ms'] for row in rows], completed, horizon)),
        'completed_latency_median_ms': float(np.median(completed_durations)) if completed_durations else None,
        'censored_duration_median_ms': float(np.median([row['duration_ms'] for row in rows if not row['completed']]))
        if len(completed_durations) < len(rows) else None,
        'horizon_ms': float(horizon),
    }


def _correction_range(rows, baselines, z=CORRECTION_Z):
    """Worst-case absolute net energy under independent batch baseline shifts."""
    if not rows:
        return None
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[row['batch_id']].append(row)
    total = len(rows)
    low = 0.0
    high = 0.0
    for batch, group in by_batch.items():
        energy = float(np.mean([_energy(row, baselines) for row in group]))
        mean_duration = float(np.mean([row['duration_ms'] for row in group]))
        shift = z * baselines[batch]['se_mw'] * mean_duration / 1000.0
        share = len(group) / total
        low += share * (energy - shift)
        high += share * (energy + shift)
    return {'low_mj': low, 'high_mj': high, 'z': z}


def _by_batch(rows, baselines, horizon):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['batch_id']].append(row)
    result = {}
    for batch, group in sorted(grouped.items()):
        metrics = _observed_metrics(group, baselines, horizon)
        metrics['correction_range'] = _correction_range(group, baselines)
        result[batch] = metrics
    return result


def _support_horizon(rows, common_blocks):
    keys = set(common_blocks)
    return max((row['duration_ms'] for row in rows if _block_key(row) in keys), default=0.0)


def _point_task_metrics(rows, baselines, config, task, common_blocks):
    keys = set(common_blocks)
    support_rows = [row for row in rows if row['config'] == config and row['task'] == task
                    and _block_key(row) in keys]
    if not support_rows:
        return None
    horizon = _support_horizon(rows, common_blocks)
    metrics = _observed_metrics(support_rows, baselines, horizon)
    metrics['correction_range'] = _correction_range(support_rows, baselines)
    metrics['by_batch'] = _by_batch(support_rows, baselines, horizon)
    metrics['support_blocks'] = len(common_blocks)
    return metrics


def _config_task_accounting(rows, baselines, config, task, common_blocks):
    all_rows = [row for row in rows if row['config'] == config and row['task'] == task]
    all_horizon = max((row['duration_ms'] for row in rows if row['task'] == task), default=0.0)
    all_metrics = _observed_metrics(all_rows, baselines, all_horizon)
    if all_metrics is not None:
        all_metrics['correction_range'] = _correction_range(all_rows, baselines)
        all_metrics['by_batch'] = _by_batch(all_rows, baselines, all_horizon)
        all_metrics['note'] = '全部记录，仅用于核对；批次和设备构成不同，不能直接用于跨配置选型。'

    support_metrics = _point_task_metrics(rows, baselines, config, task, common_blocks)
    if support_metrics is not None:
        support_metrics['note'] = '仅使用所有分析配置都测过的区组；这是跨配置比较的固定支持集。'
    return {'all_runs': all_metrics, 'common_support': support_metrics}


def _common_support(rows, configs):
    blocks = _group_blocks(rows)
    tasks = sorted({row['task'] for row in rows})
    common = {}
    excluded = {}
    for task in tasks:
        task_blocks = [key for key in blocks if key[1] == task]
        common[task] = sorted(key for key in task_blocks if all(config in blocks[key] for config in configs))
        excluded[task] = sorted(key for key in task_blocks if key not in set(common[task]))
    return blocks, common, excluded


def _point_frontier(metrics, eligible=None):
    names = sorted(metrics)
    result = []
    for name in names:
        value = metrics[name]
        dominated = False
        for other in names:
            if other == name:
                continue
            candidate = metrics[other]
            if eligible is not None and (not eligible.get(other, False) or not eligible.get(name, False)):
                continue
            if (candidate['net_energy_mj_per_task'] <= value['net_energy_mj_per_task'] + EPS
                    and candidate['latency_rmst_ms'] <= value['latency_rmst_ms'] + EPS
                    and candidate['completion_probability'] >= value['completion_probability'] - EPS
                    and (candidate['net_energy_mj_per_task'] < value['net_energy_mj_per_task'] - EPS
                         or candidate['latency_rmst_ms'] < value['latency_rmst_ms'] - EPS
                         or candidate['completion_probability'] > value['completion_probability'] + EPS)):
                dominated = True
                break
        if not dominated:
            result.append(name)
    return result


def _eligible_energy_latency_frontier(metrics, minimum):
    eligible = {name: value['completion_probability'] >= minimum - EPS for name, value in metrics.items()}
    names = sorted(metrics)
    result = []
    for name in names:
        if not eligible[name]:
            continue
        value = metrics[name]
        dominated = False
        for other in names:
            if other == name or not eligible[other]:
                continue
            candidate = metrics[other]
            if (candidate['net_energy_mj_per_task'] <= value['net_energy_mj_per_task'] + EPS
                    and candidate['latency_rmst_ms'] <= value['latency_rmst_ms'] + EPS
                    and (candidate['net_energy_mj_per_task'] < value['net_energy_mj_per_task'] - EPS
                         or candidate['latency_rmst_ms'] < value['latency_rmst_ms'] - EPS)):
                dominated = True
                break
        if not dominated:
            result.append(name)
    return result, eligible


def _joint_dominance(first, second):
    """Probability in the same bootstrap replicate, not a sum of marginal ranks."""
    return float(np.mean(
        (first[:, 0] <= second[:, 0] + EPS)
        & (first[:, 1] <= second[:, 1] + EPS)
        & (first[:, 2] >= second[:, 2] - EPS)
        & ((first[:, 0] < second[:, 0] - EPS)
           | (first[:, 1] < second[:, 1] - EPS)
           | (first[:, 2] > second[:, 2] + EPS))
    ))


def _eligible_joint_dominance(first, second, minimum):
    both = (first[:, 2] >= minimum - EPS) & (second[:, 2] >= minimum - EPS)
    return float(np.mean(both & (
        (first[:, 0] <= second[:, 0] + EPS)
        & (first[:, 1] <= second[:, 1] + EPS)
        & (first[:, 2] >= second[:, 2] - EPS)
        & ((first[:, 0] < second[:, 0] - EPS)
           | (first[:, 1] < second[:, 1] - EPS)
           | (first[:, 2] > second[:, 2] + EPS))
    )))


def _eligible_energy_latency_dominance(first, second, minimum):
    both = (first[:, 2] >= minimum - EPS) & (second[:, 2] >= minimum - EPS)
    return float(np.mean(both & (
        (first[:, 0] <= second[:, 0] + EPS)
        & (first[:, 1] <= second[:, 1] + EPS)
        & ((first[:, 0] < second[:, 0] - EPS) | (first[:, 1] < second[:, 1] - EPS))
    )))


def _conditional_eligible_energy_latency_dominance(first, second, minimum):
    """Dominance on energy/latency among replicates where both clear the gate.

    The completion threshold is treated as a gate shared by the pair, not as a
    third axis to be traded off against energy.  The probability that the pair
    is jointly eligible is returned separately so it is never hidden.
    """
    both = (first[:, 2] >= minimum - EPS) & (second[:, 2] >= minimum - EPS)
    count = int(both.sum())
    if count == 0:
        return None, 0.0
    event = ((first[:, 0] <= second[:, 0] + EPS)
             & (first[:, 1] <= second[:, 1] + EPS)
             & ((first[:, 0] < second[:, 0] - EPS) | (first[:, 1] < second[:, 1] - EPS)))
    return float(np.mean(event[both])), float(np.mean(both))


def _pair_batch_gaps(blocks, config_a, config_b, tasks, baselines):
    """Batches in which the pair lacks co-measured blocks, by task."""
    gaps = {}
    for task in tasks:
        measured = {block[2] for block in blocks if block[1] == task}
        paired = {block[2] for block in blocks
                  if block[1] == task and config_a in blocks[block] and config_b in blocks[block]}
        missing = sorted(measured - paired)
        if missing:
            gaps[task] = missing
    return gaps


def _membership(bootstrap, minimum=None):
    names = sorted(bootstrap)
    replicates = len(bootstrap[names[0]])
    dominated = np.zeros((replicates, len(names)), dtype=bool)
    feasible = np.zeros((replicates, len(names)), dtype=bool)
    eligible_dominated = np.zeros((replicates, len(names)), dtype=bool)
    for index, name in enumerate(names):
        feasible[:, index] = bootstrap[name][:, 2] >= (minimum if minimum is not None else -math.inf) - EPS
    for left in range(len(names)):
        for right in range(len(names)):
            if left == right:
                continue
            a = bootstrap[names[right]]
            b = bootstrap[names[left]]
            joint = ((a[:, 0] <= b[:, 0] + EPS)
                     & (a[:, 1] <= b[:, 1] + EPS)
                     & (a[:, 2] >= b[:, 2] - EPS)
                     & ((a[:, 0] < b[:, 0] - EPS)
                        | (a[:, 1] < b[:, 1] - EPS)
                        | (a[:, 2] > b[:, 2] + EPS)))
            dominated[:, left] |= joint
            if minimum is not None:
                eligible = ((a[:, 0] <= b[:, 0] + EPS)
                            & (a[:, 1] <= b[:, 1] + EPS)
                            & ((a[:, 0] < b[:, 0] - EPS) | (a[:, 1] < b[:, 1] - EPS)))
                # The minimum completion rate is a gate on the pair, not a third
                # ranking axis: in replicates where both clear it, compare energy
                # and latency.  Completion of the pair is reported separately.
                both_eligible = feasible[:, right] & feasible[:, left]
                eligible_dominated[:, left] |= eligible & both_eligible
    result = {name: float(np.mean(~dominated[:, index])) for index, name in enumerate(names)}
    if minimum is None:
        return result, None
    eligible_result = {}
    conditional_result = {}
    for index, name in enumerate(names):
        feasible_count = int(feasible[:, index].sum())
        if feasible_count == 0:
            conditional_result[name] = None
            eligible_result[name] = 0.0
            continue
        eligible_result[name] = float(np.mean(feasible[:, index] & ~eligible_dominated[:, index]))
        conditional_result[name] = float(np.mean(~eligible_dominated[feasible[:, index], index]))
    return result, eligible_result, conditional_result


def _block_evidence(rows, baselines, config_a, config_b, task, common_blocks):
    grouped = _group_blocks(rows)
    horizon = _support_horizon(rows, common_blocks)
    records = []
    for block in common_blocks:
        if block[1] != task or block not in grouped:
            continue
        runs_a = grouped[block].get(config_a, [])
        runs_b = grouped[block].get(config_b, [])
        if not runs_a or not runs_b:
            continue
        metrics_a = _observed_metrics(runs_a, baselines, horizon)
        metrics_b = _observed_metrics(runs_b, baselines, horizon)
        mean_duration_a = float(np.mean([row['duration_ms'] for row in runs_a]))
        mean_duration_b = float(np.mean([row['duration_ms'] for row in runs_b]))
        delta_energy = metrics_a['net_energy_mj_per_attempt'] - metrics_b['net_energy_mj_per_attempt']
        delta_latency = metrics_a['latency_rmst_ms'] - metrics_b['latency_rmst_ms']
        delta_completion = metrics_a['completion_probability'] - metrics_b['completion_probability']
        batch = block[2]
        shift = CORRECTION_Z * baselines[batch]['se_mw']
        # A - B changes by -delta_b * (mean_duration_a - mean_duration_b) / 1000.
        low_energy = delta_energy - shift * (mean_duration_a - mean_duration_b) / 1000.0
        high_energy = delta_energy + shift * (mean_duration_a - mean_duration_b) / 1000.0
        records.append({
            'device_id': block[0],
            'batch_id': batch,
            'delta_energy_mj': delta_energy,
            'delta_latency_ms': delta_latency,
            'delta_completion_probability': delta_completion,
            'energy_correction_range_mj': [min(low_energy, high_energy), max(low_energy, high_energy)],
        })
    if not records:
        return None
    energy_values = [record['delta_energy_mj'] for record in records]
    latency_values = [record['delta_latency_ms'] for record in records]
    completion_values = [record['delta_completion_probability'] for record in records]
    correction_low = float(np.mean([record['energy_correction_range_mj'][0] for record in records]))
    correction_high = float(np.mean([record['energy_correction_range_mj'][1] for record in records]))
    by_batch = {}
    for batch in sorted({record['batch_id'] for record in records}):
        batch_records = [record for record in records if record['batch_id'] == batch]
        by_batch[batch] = {
            'paired_blocks': len(batch_records),
            'mean_delta_energy_mj': float(np.mean([record['delta_energy_mj'] for record in batch_records])),
            'mean_delta_latency_ms': float(np.mean([record['delta_latency_ms'] for record in batch_records])),
            'mean_delta_completion_probability': float(np.mean(
                [record['delta_completion_probability'] for record in batch_records])),
        }
    return {
        'with_config': config_a,
        'against_config': config_b,
        'task': task,
        'paired_blocks': len(records),
        'blocks': records,
        'by_batch': by_batch,
        'mean_delta_energy_mj': float(np.mean(energy_values)),
        'energy_favours_with_blocks': int(sum(value < 0 for value in energy_values)),
        'mean_delta_latency_ms': float(np.mean(latency_values)),
        'latency_favours_with_blocks': int(sum(value < 0 for value in latency_values)),
        'mean_delta_completion_probability': float(np.mean(completion_values)),
        'completion_favours_with_blocks': int(sum(value > 0 for value in completion_values)),
        'energy_correction_range_mj': [correction_low, correction_high],
        'correction_sign_flip': (correction_low < 0 < correction_high),
    }


def _pairwise_crude_and_paired(rows, baselines, configs, common, tasks):
    blocks = _group_blocks(rows)
    result = []
    for task in tasks:
        support = set(common[task])
        for config_a, config_b in itertools.combinations(configs, 2):
            rows_a = [row for row in rows if row['task'] == task and row['config'] == config_a]
            rows_b = [row for row in rows if row['task'] == task and row['config'] == config_b]
            if not rows_a or not rows_b:
                continue
            crude_power = float(np.mean([row['power_mw'] for row in rows_a])
                                - np.mean([row['power_mw'] for row in rows_b]))
            crude_energy = float(np.mean([_energy(row, baselines) for row in rows_a])
                                 - np.mean([_energy(row, baselines) for row in rows_b]))
            paired_a = [row for row in rows_a if _block_key(row) in support]
            paired_b = [row for row in rows_b if _block_key(row) in support]
            if not paired_a or not paired_b:
                continue
            paired_energy = float(np.mean([_energy(row, baselines) for row in paired_a])
                                  - np.mean([_energy(row, baselines) for row in paired_b]))
            result.append({
                'task': task,
                'config_a': config_a,
                'config_b': config_b,
                'crude_mean_power_difference_mw_not_used_for_selection': crude_power,
                'crude_net_energy_difference_mj': crude_energy,
                'paired_net_energy_difference_mj': paired_energy,
                'paired_blocks': len(support),
                'sign_reversal_vs_crude_power': (crude_power < 0) != (paired_energy < 0),
                'sign_reversal_vs_crude_net_energy': (crude_energy < 0) != (paired_energy < 0),
            })
    return result


def _bootstrap(rows, baselines, configs, common, seed, replicates):
    grouped = _group_blocks(rows)
    rng = np.random.default_rng(seed)
    batches = sorted(baselines)
    tasks = sorted(common)
    strata = {}
    for task in tasks:
        by_batch = defaultdict(list)
        for block in common[task]:
            by_batch[block[2]].append(block)
        strata[task] = dict(by_batch)
    task_boot = {
        task: {config: np.zeros((replicates, 3), dtype=float) for config in configs}
        for task in tasks
    }
    for replicate in range(replicates):
        delta = {batch: float(rng.normal(0.0, baselines[batch]['se_mw'])) for batch in batches}
        for task in tasks:
            selected = []
            for batch, blocks in strata[task].items():
                indices = rng.integers(0, len(blocks), len(blocks))
                selected.extend(blocks[index] for index in indices)
            for config in configs:
                energies = []
                completion = []
                durations = []
                completed = []
                for block in selected:
                    runs = grouped[block][config]
                    draw = rng.integers(0, len(runs), len(runs))
                    for index in draw:
                        row = runs[index]
                        energies.append(_energy(row, baselines, delta[row['batch_id']]))
                        completion.append(1.0 if row['completed'] else 0.0)
                        durations.append(row['duration_ms'])
                        completed.append(row['completed'])
                horizon = _support_horizon(rows, common[task])
                task_boot[task][config][replicate] = [
                    float(np.mean(energies)),
                    float(_rmst(durations, completed, horizon)),
                    float(np.mean(completion)),
                ]
    return task_boot


def _workload_boot(task_boot, weights, configs):
    names = sorted(task_boot)
    result = {config: np.zeros((len(task_boot[names[0]][config]), 3), dtype=float) for config in configs}
    for config in configs:
        for index, task in enumerate(names):
            if task in weights:
                result[config][:, 0] += weights[task] * task_boot[task][config][:, 0]
                result[config][:, 1] += weights[task] * task_boot[task][config][:, 1]
                result[config][:, 2] += weights[task] * task_boot[task][config][:, 2]
    return result


def _interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _point_workload(point_tasks, config, weights):
    energy = 0.0
    latency = 0.0
    completion = 0.0
    energy_low = 0.0
    energy_high = 0.0
    for task, weight in weights.items():
        metrics = point_tasks[config][task]
        energy += weight * metrics['net_energy_mj_per_attempt']
        latency += weight * metrics['latency_rmst_ms']
        completion += weight * metrics['completion_probability']
        energy_low += weight * metrics['correction_range']['low_mj']
        energy_high += weight * metrics['correction_range']['high_mj']
    return {
        'net_energy_mj_per_task': energy,
        'latency_rmst_ms': latency,
        'completion_probability': completion,
        'net_energy_correction_range_mj': [energy_low, energy_high],
    }


def analyse(document):
    rows, baselines, workloads, minimum, seed = _validate(document)
    replicates = int(document.get('bootstrap_replicates', DEFAULT_REPLICATES))
    _require(replicates >= 100, 'bootstrap_replicates 至少为 100')

    configs = sorted({row['config'] for row in rows})
    tasks_all = sorted({row['task'] for row in rows})
    workload_tasks = sorted({task for workload in workloads for task, value in workload['weights'].items()
                             if value > 0})

    scored_configs = []
    unscored = {}
    for config in configs:
        missing = [task for task in workload_tasks
                   if not any(row['config'] == config and row['task'] == task for row in rows)]
        if missing:
            unscored[config] = {'missing_tasks': missing,
                                'reason': '未覆盖工作负载任务，不能计算完整负载指标。'}
        else:
            scored_configs.append(config)
    _require(scored_configs, '没有配置覆盖全部工作负载任务')

    blocks, common, excluded = _common_support(rows, scored_configs)
    for task in workload_tasks:
        _require(common.get(task), '任务 ' + task + ' 没有所有分析配置共同测过的区组，无法进行可比选型')

    point_tasks = {}
    for config in scored_configs:
        point_tasks[config] = {}
        for task in tasks_all:
            metrics = _point_task_metrics(rows, baselines, config, task, common.get(task, []))
            if metrics is not None:
                point_tasks[config][task] = metrics

    accounting = {'units': {
        'net_energy': 'mJ per attempt；power_mw × duration_ms / 1000，已减去同批次空载功率',
        'latency': 'ms；完成时延为右删失 RMST，未完成记录在观测停止点删失，不把 deadline_ms 当作完成时间',
        'completion': '完成概率；分母包含取消或未完成记录',
        'workload_energy': 'mJ per task；按任务权重汇总每任务净能耗，不对瓦数做跨任务平均',
    }}
    accounting['instrument_batches'] = {}
    for batch in sorted(baselines):
        batch_rows = [row for row in rows if row['batch_id'] == batch]
        accounting['instrument_batches'][batch] = {
            'baseline_power_mw': baselines[batch]['power_mw'],
            'baseline_se_mw': baselines[batch]['se_mw'],
            'runs': len(batch_rows),
            'configs': sorted({row['config'] for row in batch_rows}),
            'devices': sorted({row['device_id'] for row in batch_rows}),
            'note': 'se_mw 是同一批次共享的校正不确定性；配对比较中同一批次使用同一次偏移。',
        }
    accounting['config_task'] = {}
    for config in configs:
        accounting['config_task'][config] = {}
        for task in tasks_all:
            support = common.get(task, []) if config in scored_configs else []
            accounting['config_task'][config][task] = _config_task_accounting(
                rows, baselines, config, task, support)
    accounting['censored_attempts'] = _censoring_trace(rows)
    accounting['excluded_from_selection'] = unscored

    comparability = {
        'block_definition': '(device_id, task, batch_id)；同一区组内 repeat 是重复测量。',
        'rationale': '只在所有分析配置都出现的区组上比较，避免把批次、设备或任务构成当成配置效应；缺测区组不补造、不按负载筛选。',
        'analysis_configs': scored_configs,
        'unscored_configs': unscored,
        'tasks': {},
        'pair_coverage': [],
        'missing_pairings': [],
        'crude_vs_paired': _pairwise_crude_and_paired(rows, baselines, scored_configs, common, tasks_all),
    }
    for task in tasks_all:
        task_blocks = sorted(key for key in blocks if key[1] == task)
        comparability['tasks'][task] = {
            'total_blocks': len(task_blocks),
            'common_support_blocks': len(common.get(task, [])),
            'excluded_blocks': [
                {
                    'device_id': block[0],
                    'batch_id': block[2],
                    'present_configs': sorted(blocks[block]),
                    'missing_configs': sorted(set(scored_configs) - set(blocks[block])),
                }
                for block in excluded.get(task, [])
            ],
            'common_support_batches': {
                batch: sum(1 for block in common.get(task, []) if block[2] == batch)
                for batch in sorted(baselines)
            },
            'common_support_devices': sorted({block[0] for block in common.get(task, [])}),
            'all_config_horizon_ms': max((row['duration_ms'] for row in rows if row['task'] == task), default=0.0),
            'common_support_horizon_ms': _support_horizon(rows, common.get(task, [])),
        }
        for config_a, config_b in itertools.combinations(scored_configs, 2):
            paired = [block for block in task_blocks
                      if config_a in blocks[block] and config_b in blocks[block]]
            comparability['pair_coverage'].append({
                'task': task,
                'config_a': config_a,
                'config_b': config_b,
                'paired_blocks': len(paired),
                'all_config_support_blocks': len(common.get(task, [])),
                'blocks_by_batch': {
                    batch: sum(1 for block in paired if block[2] == batch)
                    for batch in sorted(baselines)
                },
            })
    for task in tasks_all:
        for block in excluded.get(task, []):
            missing = sorted(set(scored_configs) - set(blocks[block]))
            comparability['missing_pairings'].append({
                'task': task,
                'device_id': block[0],
                'batch_id': block[2],
                'present_configs': sorted(blocks[block]),
                'missing_configs': missing,
            })

    task_boot = _bootstrap(rows, baselines, scored_configs, common, seed, replicates)
    frontiers = {
        'method': {
            'support': '每个任务先固定为所有分析配置共同出现的区组；多组负载只改变任务权重，不改变样本。',
            'point_frontier': '三目标弱 Pareto 支配；能量和时延越低越好，完成概率越高越好。',
            'robustness': '同一区组重抽样 + 同批次共享的空载校正偏移抽样；优势概率是同一 replicate 内三目标联合事件，不是逐指标排序拼接。',
            'eligibility': '最低完成率作为硬约束；仅当该 replicate 中完成概率达到门槛才讨论其能耗与时延前沿。',
            'replicates': replicates,
            'seed': seed,
        },
        'workloads': {},
    }
    boot_by_workload = {}
    point_by_workload = {}
    for workload in workloads:
        weights, original_total = _weights(workload)
        workload_id = workload['id']
        points = {config: _point_workload(point_tasks, config, weights) for config in scored_configs}
        point_by_workload[workload_id] = points
        boot = _workload_boot(task_boot, weights, scored_configs)
        boot_by_workload[workload_id] = boot

        point_all = _point_frontier(points)
        eligible_frontier, eligible = _eligible_energy_latency_frontier(points, minimum)
        point_eligible_all = _point_frontier(points, eligible)
        membership, eligible_membership, conditional_membership = _membership(boot, minimum)

        dominance = []
        for config_a, config_b in itertools.combinations(scored_configs, 2):
            first = boot[config_a]
            second = boot[config_b]
            a_cond, joint_eligible = _conditional_eligible_energy_latency_dominance(first, second, minimum)
            b_cond, _ = _conditional_eligible_energy_latency_dominance(second, first, minimum)
            dominance.append({
                'config_a': config_a,
                'config_b': config_b,
                'a_jointly_dominates_b_probability': _joint_dominance(first, second),
                'b_jointly_dominates_a_probability': _joint_dominance(second, first),
                'a_eligible_jointly_dominates_b_probability': _eligible_joint_dominance(first, second, minimum),
                'b_eligible_jointly_dominates_a_probability': _eligible_joint_dominance(second, first, minimum),
                'a_eligible_energy_latency_dominates_b_probability_unconditional': _eligible_energy_latency_dominance(first, second, minimum),
                'b_eligible_energy_latency_dominates_a_probability_unconditional': _eligible_energy_latency_dominance(second, first, minimum),
                'pair_jointly_eligible_probability': joint_eligible,
                'a_energy_latency_dominance_probability_given_pair_eligible': a_cond,
                'b_energy_latency_dominance_probability_given_pair_eligible': b_cond,
                'diagnostic_marginal_probabilities_not_used_for_frontier': {
                    'a_energy_not_higher': float(np.mean(first[:, 0] <= second[:, 0] + EPS)),
                    'a_latency_not_higher': float(np.mean(first[:, 1] <= second[:, 1] + EPS)),
                    'a_completion_not_lower': float(np.mean(first[:, 2] >= second[:, 2] - EPS)),
                },
            })

        observations = {}
        for config in scored_configs:
            values = boot[config]
            observations[config] = {
                'net_energy_mj_per_task': points[config]['net_energy_mj_per_task'],
                'net_energy_ci95_mj_per_task': _interval(values[:, 0]),
                'net_energy_correction_range_mj': points[config]['net_energy_correction_range_mj'],
                'latency_rmst_ms': points[config]['latency_rmst_ms'],
                'latency_ci95_ms': _interval(values[:, 1]),
                'completion_probability': points[config]['completion_probability'],
                'completion_ci95': _interval(values[:, 2]),
                'feasibility_probability': float(np.mean(values[:, 2] >= minimum - EPS)),
                'pareto_membership_probability': membership[config],
                'eligible_energy_latency_membership_probability': eligible_membership[config],
                'eligible_energy_latency_membership_probability_given_self_eligible': conditional_membership[config],
                'point_eligible': bool(eligible[config]),
                'per_task_common_support': {
                    task: {
                        'net_energy_mj_per_attempt': point_tasks[config][task]['net_energy_mj_per_attempt'],
                        'latency_rmst_ms': point_tasks[config][task]['latency_rmst_ms'],
                        'completion_probability': point_tasks[config][task]['completion_probability'],
                        'attempts': point_tasks[config][task]['attempts'],
                    }
                    for task in weights if task in point_tasks[config]
                },
            }
        frontiers['workloads'][workload_id] = {
            'original_weight_sum': original_total,
            'normalised_weights': weights,
            'minimum_completion_rate': minimum,
            'support_blocks': {task: len(common[task]) for task in weights},
            'fixed_common_support_blocks': {task: len(common[task]) for task in tasks_all},
            'observations': observations,
            'point_frontier_all_three': point_all,
            'point_frontier_eligible_all_three': point_eligible_all,
            'point_frontier_eligible_energy_latency': eligible_frontier,
            'point_eligible_configs': sorted(name for name, value in eligible.items() if value),
            'joint_dominance': dominance,
        }

    classification = {'robust_keep': [], 'safe_eliminate': [], 'conditional_keep': []}

    def _safe_max(values):
        present = [value for value in values if value is not None]
        return max(present) if present else 0.0

    per_config = {}
    for config in scored_configs:
        membership_by_workload = {}
        eligible_membership_by_workload = {}
        conditional_membership_by_workload = {}
        feasibility_by_workload = {}
        point_eligible_by_workload = {}
        for workload in workloads:
            workload_id = workload['id']
            values = boot_by_workload[workload_id][config]
            observation = frontiers['workloads'][workload_id]['observations'][config]
            membership_by_workload[workload_id] = float(observation['pareto_membership_probability'])
            eligible_membership_by_workload[workload_id] = float(
                observation['eligible_energy_latency_membership_probability'])
            conditional_membership_by_workload[workload_id] = observation[
                'eligible_energy_latency_membership_probability_given_self_eligible']
            feasibility_by_workload[workload_id] = float(np.mean(values[:, 2] >= minimum - EPS))
            point_eligible_by_workload[workload_id] = bool(
                point_by_workload[workload_id][config]['completion_probability'] >= minimum - EPS)
        per_config[config] = {
            'membership_by_workload': membership_by_workload,
            'eligible_membership_by_workload': eligible_membership_by_workload,
            'conditional_membership_by_workload': conditional_membership_by_workload,
            'feasibility_by_workload': feasibility_by_workload,
            'point_eligible_by_workload': point_eligible_by_workload,
            'min_membership': min(membership_by_workload.values()),
            'max_membership': max(membership_by_workload.values()),
            'min_feasibility': min(feasibility_by_workload.values()),
        }

    dominators = {}
    for config in scored_configs:
        candidates = []
        for other in scored_configs:
            if other == config:
                continue
            per_workload = {}
            reliable = True
            for workload in workloads:
                workload_id = workload['id']
                entry = next(item for item in frontiers['workloads'][workload_id]['joint_dominance']
                             if {item['config_a'], item['config_b']} == {config, other})
                first, second = boot_by_workload[workload_id][other], boot_by_workload[workload_id][config]
                if entry['config_a'] == other:
                    dominance = entry['a_jointly_dominates_b_probability']
                    conditional = entry['a_energy_latency_dominance_probability_given_pair_eligible']
                else:
                    dominance = entry['b_jointly_dominates_a_probability']
                    conditional = entry['b_energy_latency_dominance_probability_given_pair_eligible']
                per_workload[workload_id] = {
                    'strict_three_objective_probability': dominance,
                    'energy_latency_probability_given_pair_eligible': conditional,
                    'pair_jointly_eligible_probability': entry['pair_jointly_eligible_probability'],
                    'probability_dominator_eligible_and_config_not': float(
                        np.mean((first[:, 2] >= minimum - EPS) & (second[:, 2] < minimum - EPS))),
                    'probability_config_eligible_and_dominator_not': float(
                        np.mean((second[:, 2] >= minimum - EPS) & (first[:, 2] < minimum - EPS))),
                    'probability_dominator_completion_not_lower': float(
                        np.mean(first[:, 2] >= second[:, 2] - EPS)),
                }
                if conditional is None or conditional < 0.95 - EPS:
                    reliable = False
                if per_config[other]['feasibility_by_workload'][workload_id] < 0.80 - EPS:
                    reliable = False
                if per_workload[workload_id]['probability_config_eligible_and_dominator_not'] > 0.05 + EPS:
                    reliable = False
            if reliable:
                candidates.append({'config': other, 'by_workload': per_workload})
        dominators[config] = candidates

    for config in scored_configs:
        details = per_config[config]
        record = {
            'config': config,
            'membership_by_workload': details['membership_by_workload'],
            'eligible_membership_by_workload': details['eligible_membership_by_workload'],
            'eligible_membership_by_workload_given_self_eligible': details['conditional_membership_by_workload'],
            'feasibility_by_workload': details['feasibility_by_workload'],
            'point_eligible_by_workload': details['point_eligible_by_workload'],
            'min_membership': details['min_membership'],
            'max_membership': details['max_membership'],
            'min_eligible_membership': min(details['eligible_membership_by_workload'].values()),
            'max_eligible_membership': max(details['eligible_membership_by_workload'].values()),
            'min_feasibility': details['min_feasibility'],
        }
        if (details['min_feasibility'] >= 0.80 - EPS
                and all(value is not None for value in details['conditional_membership_by_workload'].values())
                and min(details['conditional_membership_by_workload'].values()) >= 0.95 - EPS):
            record['reason'] = '在所有负载、所有同批重抽样中保持“达到完成率门槛后的能耗-时延前沿”成员资格。'
            if details['min_feasibility'] < 0.95 - EPS:
                record['caveat'] = (
                    '完成率点估计达标，但在 ' + ', '.join(
                        name for name, value in details['feasibility_by_workload'].items() if value < 0.95)
                    + ' 负载下，仍有约 '
                    + str(round(100 * (1 - details['min_feasibility']))) + '% 的同批重抽样低于最低完成率；'
                    '保留结论应随完成率门槛证据一起复查。'
                )
            classification['robust_keep'].append(record)
        elif (_safe_max(details['conditional_membership_by_workload'].values()) <= 0.05 + EPS
              and dominators[config]):
            record['dominated_by'] = dominators[config]
            record['reason'] = (
                '每个负载下都有同一配置：支配者完成率门槛概率 ≥ 0.80，被淘汰者达标而支配者不达标的概率 ≤ 0.05；'
                '在双方都达到最低完成率的同批抽样中，能耗与时延联合更优的概率 ≥ 0.95；'
                '被淘汰者进入“达标后前沿”的概率 ≤ 0.05。'
            )
            evidence = []
            for candidate in dominators[config]:
                other = candidate['config']
                task_evidence = []
                for task in sorted(common):
                    item = _block_evidence(rows, baselines, other, config, task, common[task])
                    if item is not None:
                        task_evidence.append(item)
                pair_reversals = [
                    item for item in comparability['crude_vs_paired']
                    if item['task'] in {entry['task'] for entry in task_evidence}
                    and {item['config_a'], item['config_b']} == {other, config}
                    and (item['sign_reversal_vs_crude_power'] or item['sign_reversal_vs_crude_net_energy'])
                ]
                # Missing pairings in a batch where the pair was not co-measured
                # are a real limit on extrapolation, but they do not contradict
                # the observed direction; keep them as a separate warning.
                gaps = _pair_batch_gaps(blocks, other, config, sorted(common), baselines)
                correction_flips = [item for item in task_evidence if item['correction_sign_flip']]
                batch_direction_conflicts = [
                    {'task': item['task'], 'batch_id': batch, **values}
                    for item in task_evidence
                    for batch, values in item.get('by_batch', {}).items()
                    if values['mean_delta_energy_mj'] > 0 or values['mean_delta_latency_ms'] > 0
                ]
                no_contradiction = not (pair_reversals or correction_flips or batch_direction_conflicts)
                evidence.append({
                    'dominated_by': other,
                    'by_workload': candidate['by_workload'],
                    'paired_task_evidence': task_evidence,
                    'crude_vs_paired_for_pair': pair_reversals,
                    'batches_without_pairing_for_pair': gaps,
                    'sensitivity_checks': {
                        'crude_paired_direction_reversal_for_pair': bool(pair_reversals),
                        'correction_sign_flip_tasks': [item['task'] for item in correction_flips],
                        'batch_direction_conflicts': batch_direction_conflicts,
                        'missing_pairing_batches_not_extrapolated': gaps,
                        'observed_direction_uncontradicted': no_contradiction,
                        'conclusion': (
                            '该淘汰结论不依赖单一指标排序：在全部已配对任务和批次上，配对方向与空载校正区间都不改变方向；'
                            '缺失配对批次不用于外推，已在 comparability 中保留。'
                            if no_contradiction
                            else '存在与支配方向相反的配对证据或校正区间，淘汰结论需连同这些条目复查。'
                        ),
                    },
                })
            record['evidence'] = evidence
            classification['safe_eliminate'].append(record)
        else:
            reasons = []
            if details['min_feasibility'] < 0.80 - EPS:
                reasons.append('completion_feasibility_uncertain')
            if max(details['eligible_membership_by_workload'].values()) > 0.05 + EPS:
                reasons.append('tradeoff_frontier_member')
            if _safe_max(details['conditional_membership_by_workload'].values()) > 0.05 + EPS:
                reasons.append('tradeoff_frontier_member_among_eligible_scenarios')
            if not any(details['point_eligible_by_workload'].values()):
                reasons.append('completion_below_minimum_at_point_estimate')
            if not reasons:
                reasons.append('no_single_robust_dominator')
            record['reason_codes'] = reasons
            record['reason'] = '不能放心淘汰，但也未达到稳健保留条件；需按负载和完成率风险分别看待。'
            record['workload_assessment'] = []
            for workload in workloads:
                workload_id = workload['id']
                points = point_by_workload[workload_id][config]
                record['workload_assessment'].append({
                    'workload': workload_id,
                    'point_eligible': details['point_eligible_by_workload'][workload_id],
                    'feasibility_probability': details['feasibility_by_workload'][workload_id],
                    'pareto_membership_probability': details['membership_by_workload'][workload_id],
                    'eligible_energy_latency_membership_probability':
                        details['eligible_membership_by_workload'][workload_id],
                    'net_energy_mj_per_task': points['net_energy_mj_per_task'],
                    'latency_rmst_ms': points['latency_rmst_ms'],
                    'completion_probability': points['completion_probability'],
                })
            record['closest_dominators'] = []
            for other in scored_configs:
                if other == config:
                    continue
                entry = next(item for item in frontiers['workloads'][workloads[0]['id']]['joint_dominance']
                             if {item['config_a'], item['config_b']} == {config, other})
                if entry['config_a'] == other:
                    strict = entry['a_jointly_dominates_b_probability']
                else:
                    strict = entry['b_jointly_dominates_a_probability']
                if strict > 0:
                    record['closest_dominators'].append({
                        'config': other,
                        'strict_three_objective_probability': strict,
                    })
            record['closest_dominators'].sort(key=lambda item: -item['strict_three_objective_probability'])
            record['closest_dominators'] = record['closest_dominators'][:2]
            record['what_would_resolve'] = (
                '补充该配置在缺失任务/批次上的配对重复，使完成率区间与能耗-时延优势同时变窄；'
                '若完成率区间下限仍低于最低要求，则只能作为放宽门槛时的备选。'
            )
            classification['conditional_keep'].append(record)

    critical_cases = {}
    reversals = [item for item in comparability['crude_vs_paired']
                 if item['sign_reversal_vs_crude_power'] or item['sign_reversal_vs_crude_net_energy']]
    critical_cases['paired_direction_opposite_crude_average'] = reversals
    summary_warm_gaps = [item for item in comparability['missing_pairings']
                         if item['task'] == 'summary' and item['batch_id'] == 'warm']
    critical_cases['missing_key_pairing'] = {
        'task': 'summary',
        'batch_id': 'warm',
        'blocks': summary_warm_gaps,
        'common_support_blocks_for_summary': len(common.get('summary', [])),
        'common_support_batches_for_summary': {
            batch: sum(1 for block in common.get('summary', []) if block[2] == batch)
            for batch in sorted(baselines)
        },
        'effect': 'summary 的全部跨配置结论只来自 cool 批次的共同区组；warm 下 compact/cloud 没有配对，不能把 summary 结果外推到 warm，也无法区分批次与配置交互。',
    }
    calibration_flips = []
    seen_pairs = set()
    for item in comparability['pair_coverage']:
        if item['paired_blocks'] == 0:
            continue
        pair = (item['task'], item['config_a'], item['config_b'])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        evidence = _block_evidence(rows, baselines, item['config_a'], item['config_b'],
                                   item['task'], common.get(item['task'], []))
        if evidence and evidence['correction_sign_flip']:
            calibration_flips.append(evidence)
    critical_cases['calibration_uncertainty_changes_direction'] = calibration_flips

    robustness = {
        'bootstrap': {
            'replicates': replicates,
            'seed': seed,
            'resampling_unit': '任务 × 批次内的区组（device_id, task, batch_id）有放回重抽样；同区组内 repeat 再有放回重抽样。',
            'pairing': '同一 replicate 内，同一区组的所有配置一起被抽中，保证配对；完成率、能耗、时延使用同一组重抽样样本。',
            'baseline_uncertainty': '每个 replicate 为每个批次抽取 Normal(0, se_mw) 的共享校正偏移，同一批次所有配置使用同一偏移。',
            'interval': '95% percentile interval',
            'latency_horizon': '每个任务固定为共同支持区组中的最大观测时长；超过该时长的尾部不可识别，因此不外推到 deadline_ms。',
        },
        'decision_policy': {
            'robust_keep': '所有负载下完成率门槛概率 ≥ 0.80，且“达标后前沿”成员概率 ≥ 0.95。',
            'safe_eliminate': (
                '所有负载下均存在同一支配配置：支配者完成率门槛概率 ≥ 0.80，被淘汰者达标而支配者不达标的概率 ≤ 0.05，'
                '且在双方都达标的同一批重抽样中，能耗与时延联合更优的概率 ≥ 0.95；被淘汰者进入达标后前沿的概率 ≤ 0.05。'
                '另报告不做门槛条件化的严格三目标概率，供复查。'
            ),
            'conditional_keep': '其余配置；不把“某个指标排序靠前”当作保留理由。',
            'minimum_completion_is_gate': '最低完成率是硬门槛：达标场景内比较能耗和时延，不把它折成一个可交换的指标权重。',
            'no_composite_score': '没有为三个指标设定万能权重；负载权重只决定任务混合，不决定指标之间的交换率。',
        },
        'classification': classification,
        'critical_cases': critical_cases,
        'limitations': [
            '每个任务-配置-区组通常只有 2 次重复，完成率与时延区间的精度受此限制。',
            '未完成记录的 duration_ms 尚未到达 deadline_ms；现有字段不能区分外部取消与超时，二者都按观测停止点右删失处理。',
            '设备与批次在数据中不完全交叉；同区组配对消除了设备与批次的共同偏移，但不能消除配置与批次的交互。',
            'RMST 只在观测到的时延范围内可识别，不能外推到 deadline_ms 之后。',
        ],
    }

    gap_items = []
    summary_common = len(common.get('summary', []))
    potential_summary = summary_common + len(summary_warm_gaps)
    if summary_warm_gaps:
        gap_items.append({
            'priority_class': 'common_support',
            'type': 'missing_pairing',
            'trial_class': '在 warm 的 summary 区组补齐 compact 与 cloud，并保留 local/hybrid 作为同批对照',
            'cells': summary_warm_gaps,
            'current_common_support_blocks': summary_common,
            'potential_common_support_blocks': potential_summary,
            'approximate_ci_width_factor_if_added': math.sqrt(summary_common / potential_summary),
            'why': '这是唯一同时影响 summary 的批次外推、compact/cloud 配对和 cloud 完成率判断的缺口。',
        })
    uncovered = sorted({row['device_id'] for row in rows}
                       - {row['device_id'] for row in rows if row['config'] == 'cloud'})
    if uncovered:
        gap_items.append({
            'priority_class': 'common_support',
            'type': 'device_coverage',
            'trial_class': '在未覆盖 cloud 的设备上补同任务、同批次的 cloud 配对',
            'cells': uncovered,
            'why': 'cloud 只在部分设备上出现；若要扩展到全部设备，需要设备维度的配对，而不是继续增加已有设备上的重复。',
        })

    if calibration_flips:
        affected_batches = sorted({record['batch_id']
                                   for item in calibration_flips
                                   for record in item['blocks']})
        gap_items.append({
            'priority_class': 'calibration',
            'type': 'baseline_calibration',
            'trial_class': '在同一批次内重复测量空载功耗，缩小 se_mw；优先覆盖 ' + '、'.join(affected_batches)
                          + ' 批次，并在同批次内保留成对配置',
            'cells': [
                {
                    'task': item['task'],
                    'with_config': item['with_config'],
                    'against_config': item['against_config'],
                    'paired_blocks': item['paired_blocks'],
                    'mean_delta_energy_mj': item['mean_delta_energy_mj'],
                    'energy_correction_range_mj': item['energy_correction_range_mj'],
                }
                for item in calibration_flips
            ],
            'current_baseline_se_mw': {batch: baselines[batch]['se_mw'] for batch in affected_batches},
            'why': '这些配对在 ±1.96·se 的空载校正区间内会改变能耗方向；增加配置重复无法消除该不确定性，'
                   '只有提高同批次空载测量的精度才行。',
        })

    completion_cells = []
    for config in scored_configs:
        for task in tasks_all:
            metrics = point_tasks[config].get(task)
            if metrics is None or metrics['cancelled_or_uncompleted'] == 0:
                continue
            max_weight = max((weights.get(task, 0.0) for weights in
                              [_weights(workload)[0] for workload in workloads]), default=0.0)
            completion_cells.append({
                'config': config,
                'task': task,
                'attempts': metrics['attempts'],
                'cancelled_or_uncompleted': metrics['cancelled_or_uncompleted'],
                'completion_probability': metrics['completion_probability'],
                'max_workload_task_weight': max_weight,
                'weighted_censoring': max_weight * (1.0 - metrics['completion_probability']),
            })
    completion_cells.sort(key=lambda item: (-item['weighted_censoring'], item['config'], item['task']))
    disputed = {record['config'] for record in classification['conditional_keep']}
    for cell in completion_cells:
        if cell['config'] not in disputed:
            continue
        gap_items.append({
            'priority_class': 'completion_precision',
            'type': 'completion_precision',
            'trial_class': '在 ' + cell['config'] + ' 的 ' + cell['task'] + ' 单元增加配对重复，优先覆盖当前有删失的区组',
            'cells': [cell],
            'current_attempts': cell['attempts'],
            'approximate_ci_width_factor_if_doubled': math.sqrt(cell['attempts'] / (2 * cell['attempts'])),
            'why': '该单元直接决定当前争议配置的完成率是否可靠达到门槛。',
        })
    gap_items.append({
        'priority_class': 'metadata',
        'type': 'metadata',
        'trial_class': '在后续试验中记录 stop_reason（外部取消/超时/失败）以及是否到达 deadline_ms',
        'cells': '所有未完成记录',
        'why': '现有未完成记录的 duration_ms 都低于 deadline_ms，不能从数据判断是取消还是超时；这直接影响时延删失口径的解释。',
    })

    priority_order = {'common_support': 0, 'calibration': 1, 'completion_precision': 2, 'metadata': 3}
    type_order = {'missing_pairing': 0, 'device_coverage': 1}
    gap_items.sort(key=lambda item: (priority_order[item['priority_class']],
                                     type_order.get(item['type'], 0)))
    for rank, item in enumerate(gap_items, start=1):
        item['rank'] = rank

    measurement_gaps = {
        'priority_rule': (
            '先补共同支持缺口（含设备覆盖），再补会改变方向的空载校正精度，'
            '再补决定完成率门槛的删失单元，最后补口径元数据；排序不依赖某个负载下的最有利样本。'
        ),
        'items': gap_items,
    }

    return {
        'method': {
            'analysis': '在固定共同支持区组上同时分析每任务净能耗、完成时延 RMST 和完成概率。',
            'energy_unit': 'mJ；沿用 power_mw × duration_ms / 1000 并减去同批次空载功率。',
            'latency_unit': 'ms；未完成记录在观测停止点右删失，仍进入完成概率和能耗分母。',
            'workload_unit': 'mJ per task、ms、probability；多组负载只改变任务权重。',
            'no_watt_averaging': '没有对跨任务功率做平均；负载汇总的是每任务净能耗。',
            'pairing': '同一 device/task/batch 区组内配对；共同支持集在任务层固定，所有负载共用。',
            'uncertainty': '区组配对 bootstrap 与同批次共享空载偏移联合传播；优势使用同一 replicate 的联合支配概率。',
        },
        'accounting': accounting,
        'comparability': comparability,
        'frontiers': frontiers,
        'robustness': robustness,
        'measurement_gaps': measurement_gaps,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='多目标运行配置选型分析。')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        document = read_json(args.input)
        result = analyse(document)
        write_json(args.output, result)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('处理失败：' + str(error), file=sys.stderr)
        return 2
    print('结果已保存：' + args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
