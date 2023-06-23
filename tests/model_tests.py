"""Test fluorine model"""
import unittest

import pandas as pd
from rdkit import Chem

from lefshift import constants, utils
from lefshift.fp import LEFFingerprint
from lefshift.model import FluorineModel


class ModelTests(unittest.TestCase):
    """Test fluorine model"""

    def test_predict(self):
        """Test prediction with the fluorine model"""
        smiles = "FC(F)CNC(=O)c1nc2NC=Cc2cc1 Z1625511334"
        descriptors = utils.smiles_calculate_descriptors([smiles])
        model = FluorineModel("CF3", "tests/data/test_model")
        predictions = model.predict(descriptors[constants.FP_COLUMN])
        self.assertEqual(len(predictions), 1)

    def test_confidences(self):
        """Test getting confidences and similarities with the fluorine model"""
        smiles = "FC(F)CNC(=O)c1nc2NC=Cc2cc1 Z1625511334"
        descriptors = utils.smiles_calculate_descriptors([smiles])
        model = FluorineModel("CF3", "tests/data/test_model")
        similarities, _, _ = model.confidence(descriptors[constants.FP_COLUMN])
        self.assertEqual(len(similarities), len(descriptors[constants.FP_COLUMN].values))
        similarity_neighbors = model.similarities(
            descriptors[constants.FP_COLUMN], nof_similarities=1
        )
        self.assertEqual(len(similarity_neighbors), len(descriptors[constants.FP_COLUMN].values))
        self.assertEqual(len(similarity_neighbors[0]), 1)

    def test_train(self):
        """Test the training process"""
        training_data_df = pd.DataFrame(
            [
                [
                    "Z1625511334",
                    "FC(F)CNC(=O)c1nc2NC=Cc2cc1",
                    -123.08,
                ]
            ],
            columns=[constants.ID_COLUMN, constants.SMILES_COLUMN, constants.SHIFT_COLUMN],
        )

        descriptors = utils.smiles_calculate_descriptors(
            training_data_df[constants.SMILES_COLUMN].values
        )
        training_data_df = pd.concat([training_data_df, descriptors], axis="columns")
        model = FluorineModel("CF3")
        model.train(training_data_df)
        self.assertIn("Z1625511334", model.data[constants.ID_COLUMN].values)

    def test_is_known(self):
        """Test is known fingerprint"""
        model = FluorineModel("CF3", "tests/data/test_model")
        similarity, unknown_bits = constants.SPLIT["CF3"].values()
        self.assertTrue(model.is_known(model.fingerprints[0], similarity, unknown_bits))

        mol = Chem.MolFromSmiles("S(=O)(=O)(N(CC)C)c1cncc(F)c1 Z1188305542")
        fingerprint = LEFFingerprint.generate_with_info(mol)[0][0]
        self.assertFalse(model.is_known(fingerprint, similarity, unknown_bits))
