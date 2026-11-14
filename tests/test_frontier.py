import copy
import json
import unittest
from pathlib import Path

from energy_trial_lab.frontier import analyse
from energy_trial_lab.measurements import net_energy


def load(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / name).read_text(encoding='utf-8'))


class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load('trials.json')
        cls.document['bootstrap_replicates'] = 400
        cls.result = analyse(cls.document)

    def workload(self, name):
        return self.result['frontiers']['workloads'][name]

    # --- accounting -----------------------------------------------------
    def test_energy_keeps_archive_unit_convention(self):
        row = {'batch_id': 'batch', 'power_mw': 3000, 'duration_ms': 2000, 'completed': True,
               'device_id': 'd', 'task': 'message', 'config': 'c', 'repeat': 0, 'deadline_ms': 5000}
        baselines = {'batch': {'power_mw': 500, 'se_mw': 10}}
        self.assertEqual(net_energy(row, baselines), 5000)
        self.assertNotAlmostEqual(5000, row['power_mw'] * row['duration_ms'] / 1000)

    def test_common_support_energy_matches_net_energy_formula(self):
        rows = self.document['trials']
        baselines = self.document['baselines']
        metrics = self.result['accounting']['config_task']['local']['message']['common_support']
        excluded = {(block['device_id'], block['batch_id']) for block in
                    self.result['comparability']['tasks']['message']['excluded_blocks']}
        selected = [row for row in rows if row['config'] == 'local' and row['task'] == 'message'
                    and (row['device_id'], row['batch_id']) not in excluded]
        self.assertEqual(metrics['attempts'], len(selected))
        expected = sum(net_energy(row, baselines) for row in selected) / len(selected)
        self.assertAlmostEqual(metrics['net_energy_mj_per_attempt'], expected, places=6)

    def test_no_cross_task_watt_averaging(self):
        workloads = self.workload('office')['observations']
        tasks = self.workload('office')['normalised_weights']
        recomputed = sum(weight * workloads['hybrid']['per_task_common_support'][task]['net_energy_mj_per_attempt']
                         for task, weight in tasks.items())
        self.assertAlmostEqual(workloads['hybrid']['net_energy_mj_per_task'], recomputed, places=6)

    # --- censoring ------------------------------------------------------
    def test_cancelled_attempts_stay_in_every_denominator(self):
        censored = self.result['accounting']['censored_attempts']
        self.assertTrue(censored)
        for record in censored:
            self.assertIn('completion_probability', record['counted_in'])
            self.assertIn('net_energy_mj_per_attempt', record['counted_in'])
            self.assertIn('latency_rmst_ms', record['counted_in'])
            self.assertFalse(record['reached_deadline'])
            self.assertEqual(record['stop_reason'], 'unknown; the archive does not distinguish cancellation from timeout')

    def test_latency_is_censored_rmst_not_deadline(self):
        for task in ('message', 'booking', 'summary'):
            metrics = self.result['accounting']['config_task']['compact']['summary']['all_runs']
            self.assertLessEqual(metrics['latency_rmst_ms'], metrics['horizon_ms'] + 1e-9)
            self.assertLess(metrics['latency_rmst_ms'], 5000)

    def test_censored_rows_do_not_lose_energy_or_completion(self):
        observed = self.result['accounting']['config_task']['compact']['summary']['all_runs']
        self.assertEqual(observed['attempts'], observed['completed'] + observed['cancelled_or_uncompleted'])
        self.assertLess(observed['completion_probability'], 1.0)
        self.assertGreater(observed['net_energy_mj_per_attempt'], 0)

    # --- comparability --------------------------------------------------
    def test_support_is_identical_across_workloads(self):
        office = self.workload('office')['fixed_common_support_blocks']
        travel = self.workload('travel')['fixed_common_support_blocks']
        self.assertEqual(office, travel)
        self.assertEqual(office, {'message': 10, 'booking': 10, 'summary': 5})

    def test_workload_reweighting_does_not_reselect_support(self):
        mutated = copy.deepcopy(self.document)
        mutated['workloads'].append({'id': 'summary-heavy', 'weights': {'message': 0.0, 'booking': 0.0, 'summary': 1.0}})
        result = analyse(mutated)
        self.assertEqual(result['frontiers']['workloads']['summary-heavy']['fixed_common_support_blocks'],
                         self.workload('office')['fixed_common_support_blocks'])
        self.assertEqual(result['frontiers']['workloads']['summary-heavy']['support_blocks'],
                         {'summary': 5})

    def test_missing_key_pairing_is_preserved(self):
        summary = self.result['comparability']['tasks']['summary']
        self.assertEqual(summary['common_support_batches'], {'cool': 5, 'warm': 0})
        self.assertTrue(any(block['batch_id'] == 'warm' for block in summary['excluded_blocks']))
        missing = self.result['comparability']['missing_pairings']
        warm = [item for item in missing if item['task'] == 'summary' and item['batch_id'] == 'warm']
        self.assertTrue(warm)
        self.assertTrue(all('cloud' in item['missing_configs'] or 'compact' in item['missing_configs']
                            for item in warm))
        critical = self.result['robustness']['critical_cases']['missing_key_pairing']
        self.assertEqual(critical['common_support_batches_for_summary'], {'cool': 5, 'warm': 0})
        self.assertIn('不能把 summary 结果外推到 warm', critical['effect'])

    def test_paired_direction_can_oppose_crude_average(self):
        reversals = self.result['robustness']['critical_cases']['paired_direction_opposite_crude_average']
        pairs = {(item['task'], item['config_a'], item['config_b']) for item in reversals}
        self.assertIn(('message', 'cloud', 'compact'), pairs)
        item = next(item for item in reversals
                    if (item['task'], item['config_a'], item['config_b']) == ('message', 'cloud', 'compact'))
        # crude average watts favour compact, paired net energy favours cloud.
        self.assertGreater(item['crude_mean_power_difference_mw_not_used_for_selection'], 0)
        self.assertLess(item['paired_net_energy_difference_mj'], 0)
        self.assertTrue(item['sign_reversal_vs_crude_power'])

    def test_calibration_uncertainty_can_flip_a_pair(self):
        flips = self.result['robustness']['critical_cases']['calibration_uncertainty_changes_direction']
        self.assertTrue(flips)
        item = flips[0]
        low, high = item['energy_correction_range_mj']
        self.assertLess(low, 0)
        self.assertGreater(high, 0)
        self.assertTrue(item['correction_sign_flip'])

    # --- frontiers / robustness ----------------------------------------
    def test_dominance_probability_is_joint_not_a_product_of_marginals(self):
        for workload in self.result['frontiers']['workloads'].values():
            for pair in workload['joint_dominance']:
                marginal = pair['diagnostic_marginal_probabilities_not_used_for_frontier']
                joint = pair['a_jointly_dominates_b_probability']
                self.assertLessEqual(joint, marginal['a_energy_not_higher'] + 1e-9)
                self.assertLessEqual(joint, marginal['a_latency_not_higher'] + 1e-9)
                self.assertLessEqual(joint, marginal['a_completion_not_lower'] + 1e-9)

    def test_no_config_pairs_dominates_itself(self):
        office = self.workload('office')
        configs = office['point_eligible_configs']
        self.assertEqual(sorted(configs), sorted(self.workload('travel')['point_eligible_configs']))
        names = {pair['config_a'] for pair in office['joint_dominance']} | \
                {pair['config_b'] for pair in office['joint_dominance']}
        self.assertEqual(names, set(self.result['comparability']['analysis_configs']))

    def test_classification_tells_safe_elimination_and_conditional_retention(self):
        classification = self.result['robustness']['classification']
        self.assertIn('local', [item['config'] for item in classification['safe_eliminate']])
        self.assertIn('compact', [item['config'] for item in classification['conditional_keep']])
        record = next(item for item in classification['safe_eliminate'] if item['config'] == 'local')
        self.assertTrue(record['evidence'])
        self.assertEqual(record['evidence'][0]['dominated_by'], 'hybrid')
        self.assertTrue(record['evidence'][0]['sensitivity_checks']['observed_direction_uncontradicted'])
        self.assertEqual(len(record['evidence'][0]['paired_task_evidence']), 3)

    def test_conditional_keep_reports_per_workload_tradeoff(self):
        record = next(item for item in self.result['robustness']['classification']['conditional_keep']
                      if item['config'] == 'compact')
        workloads = {item['workload'] for item in record['workload_assessment']}
        self.assertEqual(workloads, {'office', 'travel'})
        self.assertFalse(any(item['point_eligible'] for item in record['workload_assessment']))
        self.assertIn('tradeoff_frontier_member', record['reason_codes'])

    def test_analysis_is_deterministic_for_a_seed(self):
        repeated = analyse(copy.deepcopy(self.document))
        self.assertEqual(json.dumps(self.result, sort_keys=True), json.dumps(repeated, sort_keys=True))

    def test_measurement_gaps_prioritise_the_missing_pairing(self):
        items = self.result['measurement_gaps']['items']
        self.assertEqual(items[0]['type'], 'missing_pairing')
        self.assertEqual(items[0]['current_common_support_blocks'], 5)
        self.assertGreater(items[0]['potential_common_support_blocks'], 5)
        self.assertTrue(any(item['type'] == 'metadata' for item in items))
        self.assertTrue(any(item['type'] == 'completion_precision' for item in items))
        order = [item['priority_class'] for item in items]
        priority = ['common_support', 'calibration', 'completion_precision', 'metadata']
        self.assertEqual(order, sorted(order, key=priority.index))
        self.assertEqual(items[0]['type'], 'missing_pairing')
        calibration = [item for item in items if item['type'] == 'baseline_calibration']
        self.assertTrue(calibration)
        self.assertIn('message', [cell['task'] for cell in calibration[0]['cells']])
        self.assertEqual([item['rank'] for item in items], list(range(1, len(items) + 1)))

    # --- validation -----------------------------------------------------
    def test_rejects_minimum_completion_outside_unit_interval(self):
        bad = copy.deepcopy(self.document)
        bad['minimum_completion_rate'] = 1.5
        with self.assertRaises(ValueError):
            analyse(bad)

    def test_rejects_workload_task_absent_from_trials(self):
        bad = copy.deepcopy(self.document)
        bad['workloads'][0]['weights']['unknown'] = 0.1
        with self.assertRaises(ValueError):
            analyse(bad)

    def test_unscored_config_is_reported_not_silently_dropped(self):
        bad = copy.deepcopy(self.document)
        bad['trials'] = [row for row in bad['trials']
                         if not (row['config'] == 'cloud' and row['task'] == 'booking')]
        result = analyse(bad)
        self.assertIn('cloud', result['accounting']['excluded_from_selection'])
        self.assertNotIn('cloud', result['comparability']['analysis_configs'])
        self.assertIn('cloud', result['accounting']['config_task'])
