"""Fluorine chemical shift prediction"""
import argparse
import logging

from lefshift.predict import add_predict_subparser, predict
from lefshift.split import add_split_subparser, split
from lefshift.train import add_train_subparser, train

DESCRIPTION = """
LEFShift predicts fluorine chemical shifts using the local(L) environment(E)
of the fluorine(F)

examples:

Prediction only requires a SMILES column in a CSV file. Predicted values and
confidence will be appended as columns to the input and written to the output

lefshift predict input_data.csv predicted_data.csv --model /path/to/model_directory  # predict chemical shifts with a specific model
lefshift predict input_data.csv predicted_data.csv --smiles-column 'Name of SMILES column'  # specify which column contains the SMILES

Training requires an ID column a chemical shift column and a SMILES column.
The ID column is used to check whether an existing model has been trained with
this data before. Column names can be specified with their corresponding
options (e.g. "--id-column")

lefshift train training_data.csv --model /path/to/new_model_directory  # train a new model
lefshift train training_data.csv --model /path/to/new_model_directory --id-column 'Name of ID column'  # specify which column contains the ID
lefshift train training_data.csv --model /path/to/new_model_directory --shift-column 'Name of chemical shift column' # specify which column contains the ID

Out-of-distribution detection for a model can be performed with split:

lefshift split input.csv --model /path/to/new_model_directory  known.csv unknown.csv
"""


def get_args(args=None):
    """Get commandline arguments"""
    parser = argparse.ArgumentParser(
        prog="lefshift",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action")
    subparsers.required = True

    add_train_subparser(subparsers)
    add_predict_subparser(subparsers)
    add_split_subparser(subparsers)

    return parser.parse_args(args=args)


def call_subtool(args):
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.action == "predict":
        predict(args)
    elif args.action == "train":
        train(args)
    elif args.action == "split":
        split(args)
    else:
        raise RuntimeError("Invalid action")


def main():
    """Fluorine chemical shift prediction"""
    args = get_args()
    call_subtool(args)


if __name__ == "__main__":
    main()
