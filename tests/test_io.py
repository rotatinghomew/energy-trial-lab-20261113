import json
import tempfile
import unittest
from pathlib import Path
from energy_trial_lab.io import read_json, write_json


class JsonTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested/result.json'
            write_json(path, {'中文': [1, None]})
            self.assertEqual(read_json(path), {'中文': [1, None]})

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_json(Path(directory) / 'result.json', {'value': float('nan')})

    def test_duplicate_field_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.json'
            path.write_text('{"field": 1, "field": 2}')
            with self.assertRaises(ValueError):
                read_json(path)
