"""LEF constants"""
from rdkit import Chem

# CF definitions
CF_MOTIF = Chem.MolFromSmarts("[#9]-[#6]")
CF2_MOTIF = Chem.MolFromSmarts("[#9]-[#6]-[#9]")
CF3_MOTIF = Chem.MolFromSmarts("[#9]-[#6](-[#9])-[#9]")
CF_LABELS = ["CF", "CF2", "CF3"]

# Column names
BOUNDS_COLUMN = "Dice Similarity Upper Bound"
CONFIDENCE_COLUMN = "Confidence"
INTERVAL_COLUMN = "95% Confidence Interval"
ID_COLUMN = "ID"
SMILES_COLUMN = "SMILES"
SHIFT_COLUMN = "Chemical Shift"
LABEL_COLUMN = "Label"
ATOM_INDEX_COLUMN = "Atom Index"
MOL_COLUMN = "RDKit Mol"
FP_COLUMN = "Fingerprint"
FP_INFO_COLUMN = "Fingerprint Info"
PREDICTED_COLUMN = "Predicted Chemical Shift"
MAX_SIMILARITY_COLUMN = "Max Dice Similarity"
SIMILARITY_COLUMN = "Dice Similarity"

# Default model parameters
PARAMETERS = {
    "CF": {
        "colsample_bynode": 0.3,
        "learning_rate": 0.1,
        "n_estimators": 750,
    },
    "CF2": {
        "colsample_bynode": 0.5,
        "learning_rate": 0.1,
        "n_estimators": 1000,
    },
    "CF3": {
        "colsample_bynode": 0.5,
        "learning_rate": 0.1,
        "n_estimators": 750,
    },
}

# Split criteria
SPLIT = {
    "CF": {"dice_similarity": 0.8, "nof_unknown_bits": 3},
    "CF2": {"dice_similarity": 0.8, "nof_unknown_bits": 3},
    "CF3": {"dice_similarity": 0.5, "nof_unknown_bits": 2},
}
