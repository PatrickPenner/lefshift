"""Functions for training"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from lefshift import constants, utils
from lefshift.application_utils import validate_training_input
from lefshift.model import FluorineModel


def add_train_subparser(subparsers):
    """Add train arguments as a subparser"""
    train_parser = subparsers.add_parser(
        "train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument("input", help="data to train the model with", type=Path)
    train_parser.add_argument("--parameters", help="path to a parameter file", type=Path)
    # NOTE has no default model
    train_parser.add_argument(
        "-m",
        "--model",
        help="path to the directory of the model",
        type=Path,
        required=True,
    )
    train_parser.add_argument(
        "--shift-column",
        help="name of the column containing the chemical shift",
        default=constants.SHIFT_COLUMN,
        type=str,
    )
    train_parser.add_argument(
        "--id-column",
        help="name of the column containing the ID",
        default=constants.ID_COLUMN,
        type=str,
    )
    train_parser.add_argument(
        "--smiles-column",
        help="name of the column containing the molecule SMILES",
        default=constants.SMILES_COLUMN,
        type=str,
    )
    train_parser.add_argument(
        "--label-column",
        help="name of the column containing the CF/CF2/CF3 label",
        default=constants.LABEL_COLUMN,
        type=str,
    )
    train_parser.add_argument(
        "--atom-index-column",
        help="name of the column containing the atom index",
        default=constants.ATOM_INDEX_COLUMN,
        type=str,
    )
    train_parser.add_argument(
        "--cores", type=int, help="Maximum number of cores to use.", default=8
    )
    train_parser.add_argument("-v", "--verbose", help="show verbose output", action="store_true")


def prepare_model_directory(model_path):
    """Safely create the model directory

    :param model_path: path to the model directory
    :type model_path: pathlib.Path
    """
    if Path(model_path).exists():
        raise RuntimeError(f'Model directory "{model_path}" already exists')
    try:
        Path(model_path).mkdir()
    except FileNotFoundError as error:
        raise RuntimeError(f'Could not create model directory "{model_path}"') from error


def train(args):
    """Train a model"""
    training_data_df = pd.read_csv(args.input)
    training_data_df = validate_training_input(
        training_data_df, args.id_column, args.smiles_column, args.shift_column
    )
    prepare_model_directory(args.model)
    logging.info("Calculating descriptors")
    training_data_df = utils.generate_descriptors(
        training_data_df,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        label_column=args.label_column,
        atom_index_column=args.atom_index_column,
    )
    if len(training_data_df) - 1 != max(training_data_df.index):
        raise RuntimeError(
            "Training data contains molecules with different fluorines "
            "but the same chemical shift annotation"
        )

    parameters = constants.PARAMETERS
    if args.parameters is not None and Path(args.parameters).exists():
        with open(args.parameters, encoding="utf8") as args_file:
            loaded_parameters = json.load(args_file)
        for cf_label in constants.CF_LABELS:
            if cf_label not in loaded_parameters:
                continue

            logging.info("Setting %s parametrs", cf_label)
            parameters[cf_label] = loaded_parameters[cf_label]

    models = []
    for cf_label in constants.CF_LABELS:
        current_data_df = training_data_df[training_data_df[constants.LABEL_COLUMN] == cf_label]
        if len(current_data_df) == 0:
            continue

        logging.info("Training %s model", cf_label)
        current_parameters = parameters[cf_label]
        current_parameters["nthread"] = args.cores
        model = FluorineModel(cf_label, parameters=parameters[cf_label])

        model.train(
            current_data_df,
            id_column=args.id_column,
            smiles_column=args.smiles_column,
            shift_column=args.shift_column,
        )
        models.append(model)

    logging.info("Writing model to %s", args.model)
    for model in models:
        model.write(args.model)
