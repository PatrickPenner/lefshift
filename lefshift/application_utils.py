"""Utils for LEF applications"""
import logging

from lefshift import constants


def validate_column(input_df, column, data_type, error_message=None):
    """Validate that the input contains all information in valid types

    :param input_df: input dataframe
    :type input_df: pd.DataFrame
    :param column: name of the column
    :type column: str
    :param data_type: type of the data in the column
    :type data_type: type
    :param error_message: error message to display
    :type error_message: str
    :return: validated dataframe
    :rtype: pd.DataFrame
    """
    if error_message is None:
        error_message = f'Could not find column "{column}"'
    if column not in input_df.columns:
        raise RuntimeError(error_message)

    if input_df[column].isna().any():
        logging.warning('Found missing data in column "%s"', column)
    input_df = input_df.dropna(subset=[column])
    if len(input_df) == 0:
        raise RuntimeError("No valid data")

    try:
        input_df = input_df.astype(
            {
                column: data_type,
            }
        )
    except ValueError as error:
        raise RuntimeError("Input data has invalid type") from error

    return input_df


def validate_training_input(
    training_data_df,
    id_column=constants.ID_COLUMN,
    smiles_column=constants.SMILES_COLUMN,
    shift_column=constants.SHIFT_COLUMN,
):
    """Validate that the input contains all information in valid types

    :param training_data_df: input dataframe
    :type training_data_df: pd.DataFrame
    :param id_column: name of the ID column
    :type id_column: str
    :param smiles_column: name of the SMILES column
    :type smiles_column: str
    :param shift_column: name of the chemical shift column
    :type shift_column: str
    :return: validated dataframe
    :rtype: pd.DataFrame
    """
    training_data_df = validate_column(
        training_data_df,
        id_column,
        str,
        f'Could not find ID column "{id_column}" in input. Specify ID column name with "--id-column" option.',
    )
    training_data_df = validate_column(
        training_data_df,
        smiles_column,
        str,
        f'Could not find structure column "{smiles_column}" in input. Specify structure column with name "--smiles-column" option.',
    )
    training_data_df = validate_column(
        training_data_df,
        shift_column,
        float,
        f'Could not find chemical shift column "{shift_column}" in input. Specify shift column name with "--shift-column" option.',
    )

    return training_data_df
