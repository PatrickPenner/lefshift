"""Test LEF main"""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lefshift import constants
from lefshift.__main__ import get_args, call_subtool


class MainTests(unittest.TestCase):
    """Test LEF main"""

    def test_predict(self):
        """Test prediction"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = "tests/data/test_data.csv"
            output_path = tmp_path / "output.csv"
            similarities_path = tmp_path / "similarities.csv"
            args = get_args(
                [
                    "predict",
                    input_path,
                    str(output_path),
                    "--model",
                    "tests/data/test_model/",
                    "--smiles-column",
                    "SMILES",
                    "--similarities",
                    str(similarities_path),
                ]
            )
            call_subtool(args)
            output = pd.read_csv(output_path)
            similarities = pd.read_csv(similarities_path)
            test_data = pd.read_csv(input_path)
            self.assertEqual(len(output), len(test_data))
            self.assertEqual(len(test_data.columns) + 4, len(output.columns))
            self.assertEqual(len(test_data.columns) + 4, len(similarities.columns))

    def test_train(self):
        """Test training"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = "tests/data/test_data.csv"
            model_path = tmp_path / "model"
            # model_path = Path("tests/data/test_model")
            parameters_path = tmp_path / "parameters.json"
            with open(parameters_path, "w", encoding="utf8") as parameter_file:
                json.dump(constants.PARAMETERS, parameter_file)
            args = get_args(
                [
                    "train",
                    input_path,
                    "--model",
                    str(model_path),
                    "--id-column",
                    "ID",
                    "--smiles-column",
                    "SMILES",
                    "--shift-column",
                    "Shift 1 (ppm)",
                    "--parameters",
                    str(parameters_path),
                    "--cores",
                    "1",
                ]
            )
            call_subtool(args)
            self.assertTrue(model_path.exists())
            self.assertEqual(len(list(model_path.glob("*.bin"))), 3)
            self.assertEqual(len(list(model_path.glob("*.csv"))), 3)
            self.assertEqual(len(list(model_path.glob("*.json"))), 3)

    def test_split(self):
        """Test splitting"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = "tests/data/test_data.csv"
            known_path = tmp_path / "known.csv"
            unknown_path = tmp_path / "unknown.csv"
            args = get_args(
                [
                    "split",
                    input_path,
                    "--model",
                    "tests/data/test_model/",
                    str(known_path),
                    str(unknown_path),
                ]
            )
            call_subtool(args)
            known = pd.read_csv(known_path)
            unknown = pd.read_csv(unknown_path)
            self.assertEqual(len(known), 3)
            # 2 fluorines from the ChEMBL compound are unknown
            self.assertEqual(len(unknown), 2)
