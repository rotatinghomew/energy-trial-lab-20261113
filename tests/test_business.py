import copy
import json
import tempfile
import unittest
from pathlib import Path


def load(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / name).read_text(encoding='utf-8'))


from energy_trial_lab.measurements import net_energy, trial_keys, inspect


class BusinessTests(unittest.TestCase):
    def setUp(self):
        self.document = load('trials.json')

    def test_milli_units(self):
        row = {'batch_id': 'batch', 'power_mw': 3000, 'duration_ms': 2000}
        self.assertEqual(net_energy(row, {'batch': {'power_mw': 500, 'se_mw': 20}}), 5000)

    def test_cancellation_retained(self):
        result = inspect(self.document)['observed']
        self.assertEqual(sum(value['runs'] for value in result.values()), len(self.document['trials']))
        self.assertTrue(any(value['completion_rate'] < 1 for value in result.values()))

    def test_negative_net_is_visible(self):
        row = {'batch_id': 'batch', 'power_mw': 50, 'duration_ms': 1000}
        self.assertEqual(net_energy(row, {'batch': {'power_mw': 100, 'se_mw': 10}}), -50)

    def test_duplicate_run(self):
        row = self.document['trials'][0]
        with self.assertRaises(ValueError):
            trial_keys([row, row])

    def test_bad_measurement(self):
        row = copy.deepcopy(self.document['trials'][0])
        row['duration_ms'] = -1
        with self.assertRaises(ValueError):
            net_energy(row, self.document['baselines'])
