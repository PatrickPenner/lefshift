"""Fluorine chemical shift prediction model"""
import json
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import BulkDiceSimilarity
from xgboost import XGBRegressor

from lefshift import constants, utils
from lefshift.fp import LEFFingerprint


class FluorineModel:
    """Fluorine chemical shift prediction model"""

    def __init__(self, label, model_dir=None, parameters=None):
        """Fluorine chemical shift prediction model

        :param label: CF label of the model to load
        :type label: str
        :param model_dir: directory to load model from
        :type model_dir: pathlib.Path
        :parma parameters: parameters to use for the model when training
        :type parameters: dict
        """
        self.label = label
        self.model_dir = model_dir
        self.parameters = parameters
        self.model = self.load_model()

        self.data = None
        self.mols = None
        self.fingerprints = None
        self.bound = None
        self.intervals = None
        if self.model_dir is not None:
            self.data, self.mols, self.fingerprints, self.bit_set = self.load_data()
            self.bounds, self.confidences, self.intervals = self.load_bounds()

    def train(
        self,
        training_data_df,
        id_column=constants.ID_COLUMN,
        smiles_column=constants.SMILES_COLUMN,
        shift_column=constants.SHIFT_COLUMN,
    ):
        """Train the fluorine model

        :param training_data_df: training data
        :type training_data_df: pd.DataFrame
        :param id_column: name of the column containing the ID
        :type id_column: str
        :param smiles_column: name of the column containing the structure
        :type smiles_column: str
        :param shift_column: name of the chemical shift column
        :type shift_column: str
        """
        # normalize training data
        training_data_df = training_data_df.rename(
            columns={
                id_column: constants.ID_COLUMN,
                smiles_column: constants.SMILES_COLUMN,
                shift_column: constants.SHIFT_COLUMN,
            }
        )
        training_data_df.drop_duplicates(subset=[constants.ID_COLUMN])

        if self.data is not None:
            self.data = pd.concat([self.data, training_data_df], axis="rows")
        else:
            self.data = training_data_df

        if self.mols is not None:
            self.mols.extend(training_data_df[constants.MOL_COLUMN])
        else:
            self.mols = training_data_df[constants.MOL_COLUMN]

        if self.fingerprints is not None:
            self.fingerprints.extend(training_data_df[constants.FP_COLUMN])
        else:
            self.fingerprints = training_data_df[constants.FP_COLUMN]

        fp_batch = [fp.ToList() for fp in training_data_df[constants.FP_COLUMN]]
        self.model.fit(fp_batch, training_data_df[constants.SHIFT_COLUMN])

    def predict(self, lef_fingerprints):
        """Predict fluorine chemical shifts with LEFFingerprints

        :param lef_fingerprints: list of LEFFingerprints
        :type lef_fingerprints: list[rdkit.DataStructs.cDataStructs.LongSparseIntVect]
        :return: list of chemical shift predictions
        :rtype: list[float]
        """
        logging.info('Prediction with model "%s"', self.label)
        fp_batch = [fp.ToList() for fp in lef_fingerprints]
        return self.model.predict(fp_batch)

    def confidence(self, lef_fingerprints):
        """Extract confidences for LEFFingerprints

        :param lef_fingerprints: list of LEFFingerprints
        :type lef_fingerprints: list[rdkit.DataStructs.cDataStructs.LongSparseIntVect]
        :return: list of similarities, list of confidences and list of confidence intervals
        :rtype: list[float] or None, list[float] or None, list[float] or None
        """
        logging.info('Extracting confidences with model "%s"', self.label)
        if self.data is None:
            logging.warning("Could not generate confidence information without training data")
            return None, None, None

        similarities = []
        for fingerprint in lef_fingerprints:
            similarities.append(utils.highest_dice_similarity(fingerprint, self.fingerprints)[0])

        if self.bounds is None:
            logging.warning(
                "Could not generate confidence interval information without bounds data"
            )
            return similarities, None, None

        confidences = []
        intervals = []
        for similarity in similarities:
            index = utils.hist_index(similarity, self.bounds)
            confidences.append(self.confidences[index])
            intervals.append(self.intervals[index])
        return similarities, confidences, intervals

    def similarities(self, lef_fingerprints, nof_similarities=100):
        """Generate similarities for LEFFingerprints to the training data

        :param lef_fingerprints: list of LEFFingerprints
        :type lef_fingerprints: list[rdkit.DataStructs.cDataStructs.LongSparseIntVect]
        :return: list of tuples of similarity and index in the training data set
        :rtype: list[tuple[float, int]] or None
        """
        logging.info('Generating similarity information with model "%s"', self.label)
        if self.data is None:
            logging.warning("Could not generate similarity information without training data")
            return None

        similarities = []
        for fingerprint in lef_fingerprints:
            neighbors = utils.sorted_dice_neighbors(fingerprint, self.fingerprints)
            similarities.append(neighbors[:nof_similarities])
        return similarities

    def is_known(self, fingerprint, min_similarity, max_nof_unknown):
        """Are the fingerprints features known by the model

        :param fingerprint: fingerprint to check
        :type fingerprint: rdkit.DataStructs.cDataStructs.LongSparseIntVect
        :param min_similarity: minimum similarity to be considered known
        :type min_similarity: float
        :param max_nof_unknown: number of unknown to be considered unknown
        :type max_nof_unknown: int
        :return: is the fingerprint known
        :rtype: bool
        """
        max_similarity = max(BulkDiceSimilarity(fingerprint, self.fingerprints))
        if max_similarity < min_similarity:
            return False

        nof_unknown = utils.bit_difference(fingerprint, self.bit_set)
        if nof_unknown >= max_nof_unknown:
            return False

        return True

    def load_model(self):
        """Load model

        Models are loaded from the model directory as pickle files named label + ".pickle"
        """
        if self.model_dir is not None:
            logging.info('Loading model from "%s" with label "%s"', self.model_dir, self.label)
            model_path = Path(self.model_dir) / (self.label + ".bin")
            if not model_path.exists():
                # is an exception because it is program critical
                raise RuntimeError(f'Could not find model "{model_path}"')

            model = XGBRegressor()
            model.load_model(model_path)
        else:
            logging.info("Initializing empty model for training")
            parameters = self.parameters
            if parameters is None:
                parameters = {}

            model = XGBRegressor(**parameters)
        return model

    def load_data(self):
        """Load training data

        Training data is loaded from the model directory as
        label + "_training.csv". It is assumed to have an ID_COLUMN,
        a SMILES_COLUMN, and a SHIFT_COLUMN.
        """
        data_path = Path(self.model_dir) / (self.label + "_training.csv")
        if not data_path.exists():
            # is a logging error because it indicates an improper models directory
            logging.error('Could not find data "%s"', data_path)
            return None, None, None, None

        training_data_df = pd.read_csv(data_path)
        mols = [
            Chem.MolFromSmiles(smiles)
            for smiles in training_data_df[constants.SMILES_COLUMN].values
        ]
        atom_indexes = training_data_df[constants.ATOM_INDEX_COLUMN].values
        fingerprints = [
            LEFFingerprint.generate_with_info(mol, from_atoms=[[int(atom_index)]])[0][0]
            for mol, atom_index in zip(mols, atom_indexes)
        ]

        return training_data_df, mols, fingerprints, utils.get_bit_set(fingerprints)

    def load_bounds(self):
        """Load error bounds

        Bounds files are assumed to have a BOUNDS_COLUMN, a CONFIDENCE_COLUMN,
        and an INTERVAL_COLUMN.
        """
        bounds_path = Path(self.model_dir) / (self.label + "_bounds.csv")
        if not bounds_path.exists():
            # is a logging warning because it is involved in the output, although not critically
            logging.info('Could not find bounds data "%s"', bounds_path)
            return None, None, None

        bounds_df = pd.read_csv(bounds_path)
        if constants.BOUNDS_COLUMN not in bounds_df:
            logging.warning(
                'Bounds file "%s" does not contain bounds column "%s"',
                bounds_path,
                constants.BOUNDS_COLUMN,
            )
            return None, None, None

        if constants.CONFIDENCE_COLUMN not in bounds_df:
            logging.warning(
                'Bounds file "%s" does not contain confidence column "%s"',
                bounds_path,
                constants.CONFIDENCE_COLUMN,
            )
            return None, None, None

        if constants.INTERVAL_COLUMN not in bounds_df:
            logging.warning(
                'Bounds file "%s" does not contain interval column "%s"',
                bounds_path,
                constants.INTERVAL_COLUMN,
            )
            return None, None, None

        return (
            bounds_df[constants.BOUNDS_COLUMN].values,
            bounds_df[constants.CONFIDENCE_COLUMN].values,
            bounds_df[constants.INTERVAL_COLUMN].values,
        )

    def write(self, model_dir):
        """Write model to a directory

        :param model_dir: path to the model directory
        :type model_path: str
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            raise RuntimeError(f'Could not find directory "{model_dir}"')

        self.model.save_model(model_path / (self.label + ".bin"))

        # only save relevant data
        self.data = self.data[
            [
                constants.ID_COLUMN,
                constants.SMILES_COLUMN,
                constants.SHIFT_COLUMN,
                constants.LABEL_COLUMN,
                constants.ATOM_INDEX_COLUMN,
            ]
        ]
        self.data.to_csv(model_path / (self.label + "_training.csv"), index=False)

        if self.bound is not None:
            bounds_df = pd.DataFrame(
                zip(self.bounds, self.intervals),
                columns=[constants.BOUNDS_COLUMN, constants.INTERVAL_COLUMN],
            )
            bounds_df.to_csv(model_path / (self.label + "_bounds.csv"), index=False)

        if self.parameters is not None:
            with open(
                model_path / (self.label + "_parameters.json"), "w", encoding="utf8"
            ) as parameter_file:
                json.dump({self.label: self.parameters}, parameter_file)
