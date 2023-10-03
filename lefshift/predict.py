"""Functions for prediction"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lefshift import constants, utils
from lefshift.application_utils import validate_column
from lefshift.model import FluorineModel


def add_predict_subparser(subparsers):
    """Add predict arguments as a subparser"""
    predict_parser = subparsers.add_parser(
        "predict", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict_parser.add_argument("input", help="input to predict chemical shifts for", type=Path)
    predict_parser.add_argument(
        "output",
        help="predicted chemical shift information appended to the input file",
        type=Path,
    )
    predict_parser.add_argument(
        "-m",
        "--model",
        help="path to the directory of the model",
        type=Path,
        required=True,
    )
    predict_parser.add_argument(
        "-s",
        "--similarities",
        help="write similarities of the input to the training data",
        type=Path,
    )
    predict_parser.add_argument(
        "--nof-similarities",
        help="number of similarities per input record to write",
        default=10,
        type=int,
    )
    predict_parser.add_argument(
        "--smiles-column",
        help="name of the column containing the molecule SMILES",
        default=constants.SMILES_COLUMN,
        type=str,
    )
    predict_parser.add_argument(
        "--label-column",
        help="name of the column containing the CF/CF2/CF3 label",
        default=constants.LABEL_COLUMN,
        type=str,
    )
    predict_parser.add_argument(
        "--atom-index-column",
        help="name of the column containing the atom index",
        default=constants.ATOM_INDEX_COLUMN,
        type=str,
    )
    predict_parser.add_argument("-v", "--verbose", help="show verbose output", action="store_true")


def make_predictions(descriptors, model):
    """Make predictions

    Also package these predictions into a DataFrame

    :param descriptors: fluorine descriptors
    :type descriptors: pd.DataFrame
    :param model: model to predict with
    :type model: FluorineModel
    :return: data frame with predictions and confidence information
    :rtype: pd.DataFrame
    """
    predictions = model.predict(descriptors[constants.FP_COLUMN])
    similarities, confidences, intervals = model.confidence(descriptors[constants.FP_COLUMN])
    if similarities is None:
        similarities = np.full(len(predictions), np.NaN, dtype=np.float64)
    if confidences is None:
        confidences = np.full(len(predictions), np.NaN, dtype=np.float64)
    if intervals is None:
        intervals = np.full(len(predictions), np.NaN, dtype=np.float64)
    rows = list(zip(predictions, similarities, confidences, intervals))
    return pd.DataFrame(
        rows,
        columns=[
            constants.PREDICTED_COLUMN,
            constants.MAX_SIMILARITY_COLUMN,
            constants.CONFIDENCE_COLUMN,
            constants.INTERVAL_COLUMN,
        ],
        index=descriptors.index,
    )


def make_similarities(descriptors, model, nof_similarities):
    """Generate similarity information

    :param descriptors: fluorine descriptors
    :type descriptors: pd.DataFrame
    :param model: model to generate similarities for
    :type model: FluorineModel
    :param nof_similarities: number of top n similarities to extract
    :type nof_similarities: int
    :return: data frame with similarity information
    :rtype: pd.DataFrame or None
    """
    similarities = model.similarities(
        descriptors[constants.FP_COLUMN], nof_similarities=nof_similarities
    )
    if similarities is None:
        return None

    model_data = model.data[[constants.ID_COLUMN, constants.SMILES_COLUMN, constants.SHIFT_COLUMN]]
    rows = []
    for index, input_index in enumerate(descriptors.index):
        for similarity in similarities[index]:
            rows.append([input_index, *similarity])

    similarities_df = pd.DataFrame(
        rows, columns=["Index", constants.SIMILARITY_COLUMN, "Training Index"]
    )
    similarities_df = (
        similarities_df.set_index("Training Index").join(model_data).set_index("Index")
    )
    return similarities_df


def write_output(input_df, output_df, output_path):
    """Write output"""
    suffix = ""
    if any(column for column in output_df.columns if column in set(input_df.columns)):
        suffix = " Calculated"
    drop_columns = [constants.MOL_COLUMN, constants.FP_COLUMN]
    if constants.FP_INFO_COLUMN in input_df.columns:
        drop_columns.append(constants.FP_INFO_COLUMN)
    output_df = input_df.join(output_df, rsuffix=suffix).drop(drop_columns, axis="columns")
    output_df.to_csv(output_path, index=False)


def write_similarities(input_df, similarities_df, similarities_path):
    """Write similarities"""
    suffix = ""
    if any(column for column in similarities_df if column in set(input_df.columns)):
        suffix = " Training Data"
    drop_columns = [constants.MOL_COLUMN, constants.FP_COLUMN]
    if constants.FP_INFO_COLUMN in input_df.columns:
        drop_columns.append(constants.FP_INFO_COLUMN)
    similarities_df = input_df.join(similarities_df, rsuffix=suffix).drop(
        drop_columns, axis="columns"
    )
    similarities_df.to_csv(similarities_path, index=False)


def predict(args):
    """Predict chemical shifts"""
    if not Path(args.model).exists():
        raise RuntimeError(f'Could not find model directory "{args.model}"')

    input_df = pd.read_csv(args.input)
    input_df = validate_column(
        input_df,
        args.smiles_column,
        str,
        f'Could not find structure column "{args.smiles_column}" in input. Specify structure column with name "--smiles-column" option.',
    )

    logging.info("Calculating descriptors")
    input_df = utils.generate_descriptors(
        input_df,
        smiles_column=args.smiles_column,
        label_column=args.label_column,
        atom_index_column=args.atom_index_column,
    )
    input_df = input_df.reset_index(drop=True)

    output = []
    similarities = []
    for cf_label in constants.CF_LABELS:
        current_input = input_df[input_df[constants.LABEL_COLUMN] == cf_label]
        if len(current_input) == 0:
            continue

        logging.info("%s prediction", cf_label)
        try:
            model = FluorineModel(cf_label, args.model)
        except RuntimeError as error:
            logging.warning(error)
            continue

        output.append(make_predictions(current_input, model))
        if args.similarities is not None:
            logging.info("Collecting %s similarities", cf_label)
            similarity_group = make_similarities(current_input, model, args.nof_similarities)
            if similarity_group is not None:
                similarities.append(similarity_group)

    output_df = pd.concat(output, axis="rows")
    write_output(input_df, output_df, args.output)
    if args.similarities is not None:
        similarities_df = pd.concat(similarities, axis="rows")
        write_similarities(input_df, similarities_df, args.similarities)
