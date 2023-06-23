"""Test application utils"""
import unittest

import pandas as pd

from lefshift import constants
from lefshift.application_utils import validate_column, validate_training_input


class ApplicationUtilsTests(unittest.TestCase):
    """Test application utils"""

    def test_validate_input(self):
        """Test input validation"""
        empty_df = pd.DataFrame([])
        self.assertRaises(
            RuntimeError,
            validate_column,
            empty_df,
            constants.SMILES_COLUMN,
            str,
            f'Could not find structure column "{constants.SMILES_COLUMN}" in input.'
            ' Specify structure column with name "--smiles-column" option.',
        )

        nan_df = pd.DataFrame([None], columns=[constants.SMILES_COLUMN])
        self.assertRaises(RuntimeError, validate_column, nan_df, constants.SMILES_COLUMN, str)

        nan_df = pd.DataFrame(["Invalid"], columns=[constants.SHIFT_COLUMN])
        self.assertRaises(RuntimeError, validate_column, nan_df, constants.SHIFT_COLUMN, float)

    def test_validate_training_input(self):
        """Test training input validation"""
        training_data_df = pd.DataFrame(
            [["Z1188305542", "S(=O)(=O)(N(CC)C)c1cncc(F)c1", -122.67]],
            columns=[constants.ID_COLUMN, constants.SMILES_COLUMN, constants.SHIFT_COLUMN],
        )
        self.assertIsNotNone(validate_training_input(training_data_df))
