"""Split input into known and unknown feature sets"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lefshift import constants, utils
from lefshift.application_utils import validate_column
from lefshift.model import FluorineModel


def add_split_subparser(subparsers):
    """Add split arguments as a subparser"""
    split_parser = subparsers.add_parser(
        "split", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    split_parser.add_argument("input", help="input CSV to split", type=Path)
    split_parser.add_argument(
        "-m",
        "--model",
        help="path to the directory of the model",
        type=Path,
        required=True,
    )
    split_parser.add_argument("known", help="output CSV of samples with known features", type=Path)
    split_parser.add_argument(
        "unknown", help="output CSV of samples with unknown features", type=Path
    )
    split_parser.add_argument(
        "--smiles-column",
        help="name of the column containing the molecule SMILES",
        default=constants.SMILES_COLUMN,
        type=str,
    )
    split_parser.add_argument("-v", "--verbose", help="show verbose output", action="store_true")


def split(args):
    """Split input into known and unknown feature sets"""

    if not Path(args.model).exists():
        raise RuntimeError(f'Could not find model directory "{args.model}"')

    input_df = pd.read_csv(args.input)
    input_df = validate_column(
        input_df,
        args.smiles_column,
        str,
        f'Could not find structure column "{args.smiles_column}" in input. Specify structure column with name "--smiles-column" option.',
    )

    split_criteria = constants.SPLIT

    logging.info("Calculating descriptors")
    if "Label" in input_df.columns and "Atom Index" in input_df.columns:
        input_df = validate_column(input_df, "Label", str)
        input_df = validate_column(input_df, "Atom Index", int)
        descriptors = input_df.join(utils.calculate_fingerprints(input_df, args.smiles_column))
    else:
        descriptors = input_df.join(
            utils.smiles_calculate_descriptors(input_df[args.smiles_column])
        )

    known = []
    unknown = []
    for cf_label in constants.CF_LABELS:
        current_descriptors = descriptors[descriptors[constants.LABEL_COLUMN] == cf_label]
        if len(current_descriptors) == 0:
            continue

        logging.info("%s splitting", cf_label)
        try:
            model = FluorineModel(cf_label, args.model)
            min_similarity = split_criteria[cf_label]["dice_similarity"]
            max_nof_unknown = split_criteria[cf_label]["nof_unknown_bits"]
            is_known = [
                model.is_known(
                    fingerprint, min_similarity=min_similarity, max_nof_unknown=max_nof_unknown
                )
                for fingerprint in current_descriptors[constants.FP_COLUMN]
            ]
            known.append(current_descriptors[is_known])
            unknown.append(current_descriptors[~np.array(is_known)])
        except RuntimeError as error:
            logging.warning(error)
            continue

    if len(known) != 0:
        known_df = pd.concat(known)
        drop_columns = [constants.MOL_COLUMN, constants.FP_COLUMN]
        if constants.FP_INFO_COLUMN in known_df.columns:
            drop_columns.append(constants.FP_INFO_COLUMN)
        known_df = known_df.drop(drop_columns, axis="columns")
        logging.info("%s known", len(known_df))
        known_df.to_csv(args.known, index=False)

    if len(unknown) != 0:
        unknown_df = pd.concat(unknown)
        drop_columns = [constants.MOL_COLUMN, constants.FP_COLUMN]
        if constants.FP_INFO_COLUMN in unknown_df.columns:
            drop_columns.append(constants.FP_INFO_COLUMN)
        unknown_df = unknown_df.drop(drop_columns, axis="columns")
        logging.info("%s unknown", len(unknown_df))
        unknown_df.to_csv(args.unknown, index=False)
