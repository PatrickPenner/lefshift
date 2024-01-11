"""Utils"""
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS, SmilesParserParams
from rdkit.Chem.rdchem import ChiralType
from rdkit.DataStructs import BulkDiceSimilarity

from lefshift import constants
from lefshift.fp import LEFFingerprint


def nof_unique_fingerprints(mol):
    """Get number of unique LEF fingerprints

    :param mol: molecule to calculate fingerprints for
    :type mol: rdkit.Chem.rdchem.Mol
    :return: number of unique LEF fingerprints
    :rtype: int
    """
    fingerprints, _ = LEFFingerprint.generate_with_info(mol)
    return len({str(fingerprint.GetNonzeroElements()) for fingerprint in fingerprints})


def smiles_nof_unique_fingerprints(smiles):
    """Does the SMILES have the exact nof unique LEF fingreprints

    Uses the nonzero fingeprint elements and the ordered natuer of a python dict.

    :param smiles: SMILES to check
    :type smiles: str
    :return: number of unique LEF fingerprints
    :rtype: int
    """
    params = SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    return nof_unique_fingerprints(mol)


def highest_dice_similarity(fingerprint, fingerprints):
    """Return highest similarity of a fingerprint in the set to the fingerprint

    :param fingerprint: query fingerprint
    :type fingerprint: rdkit.DataStructs.cDataStructs.LongSparseIntVect
    :param fingerprints: fingerprints to find similarity to
    :type fingerprints: list[rdkit.DataStructs.cDataStructs.LongSparseIntVect]
    :return: highest similarity its index
    :rtype: float, int
    """
    similarities = BulkDiceSimilarity(fingerprint, fingerprints)
    highest_similarity = None
    highest_similarity_index = None
    for i, similarity in enumerate(similarities):
        if highest_similarity is None or similarity > highest_similarity:
            highest_similarity = similarity
            highest_similarity_index = i
    return highest_similarity, highest_similarity_index


def sorted_dice_neighbors(fingerprint, fingerprints):
    """Return neighbors in dice space sorted by similarity

    :param fingerprint: query fingerprint
    :type fingerprint: rdkit.DataStructs.cDataStructs.LongSparseIntVect
    :param fingerprints: fingerprints to find similarity to
    :type fingerprints: list[rdkit.DataStructs.cDataStructs.LongSparseIntVect]
    :return: nearest neighbors in dice space as a tuple of similarity and index
    :rtype: list[tuple[float, int]]
    """
    similarities = BulkDiceSimilarity(fingerprint, fingerprints)
    return sorted(zip(similarities, range(len(similarities))), key=lambda x: x[0], reverse=True)


def has_smarts(smiles, smarts):
    """SMILES contains SMARTS expression

    :param smiles: SMILES string of the molecule to test
    :type smiles: str
    :param smarts: SMARTS string to check for
    :type smarts: str
    :return: SMILES contains SMARTS
    :rtype: bool
    """
    smarts_graph = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    return mol.HasSubstructMatch(smarts_graph)


def get_cf_matches(mol):
    """Get matches of CF moiety

    CF2 and CF3 do not count as CF even though they are supersets

    :param mol: molecule to test
    :type mol: rdkit.Chem.rdchem.Mol
    :return: CF matches
    :rtype: list[list[int]]
    """
    cf_matches = mol.GetSubstructMatches(constants.CF_MOTIF)
    cf2_matches = mol.GetSubstructMatches(constants.CF2_MOTIF)
    cf3_matches = mol.GetSubstructMatches(constants.CF3_MOTIF)
    cf_matches = [
        match
        for match in cf_matches
        if any(
            index
            for index in match
            if not any(cf2_match for cf2_match in cf2_matches if index in set(cf2_match))
            and not any(cf3_match for cf3_match in cf3_matches if index in set(cf3_match))
        )
    ]
    return cf_matches


def has_cf(mol):
    """Mol contains CF moiety

    CF2 and CF3 do not count as CF even though they are supersets

    :param mol: molecule to test
    :type mol: rdkit.Chem.rdchem.Mol
    :return: Mol contains CF
    :rtype: bool
    """
    return any(get_cf_matches(mol))


def smiles_has_cf(smiles):
    """SMILES contains CF moiety

    :param smiles: SMILES string of molecule to test
    :type smiles: str
    :return: SMILES contains CF
    :rtype: bool
    """
    mol = Chem.MolFromSmiles(smiles)
    return has_cf(mol)


def get_cf2_matches(mol):
    """Get matches of CF2 moiety

    CF3 does not count as CF2 even though it is a superset

    :param mol: molecule to test
    :type mol: rdkit.Chem.rdchem.Mol
    :return: CF2 matches
    :rtype: list[list[int]]
    """
    cf2_matches = mol.GetSubstructMatches(constants.CF2_MOTIF)
    cf3_matches = mol.GetSubstructMatches(constants.CF3_MOTIF)
    cf2_matches = [
        match
        for match in cf2_matches
        if any(
            index
            for index in match
            if not any(cf3_match for cf3_match in cf3_matches if index in set(cf3_match))
        )
    ]
    return cf2_matches


def has_cf2(mol):
    """Mol contains CF2 moiety

    CF3 does not count as CF2 even though it is a superset

    :param mol: molecule to test
    :type mol: rdkit.Chem.rdchem.Mol
    :return: Mol contains CF2
    :rtype: bool
    """
    return any(get_cf2_matches(mol))


def smiles_has_cf2(smiles):
    """SMILES contains CF2 moiety

    :param smiles: SMILES string of molecule to test
    :type smiles: str
    :return: SMILES contains CF2
    :rtype: bool
    """
    mol = Chem.MolFromSmiles(smiles)
    return has_cf2(mol)


def get_cf3_matches(mol):
    """Get matches of CF3 moiety

    :param mol: molecule to test
    :type mol: rdkit.Chem.rdchem.Mol
    :return: CF3 matches
    :rtype: list[list[int]]
    """
    return mol.GetSubstructMatches(constants.CF3_MOTIF)


def has_cf3(mol):
    """Mol contains CF3 moiety

    :param mol: molecule to test
    :type mol: rdkit.Chem.rdchem.Mol
    :return: Mol contains CF3
    :rtype: bool
    """
    return any(get_cf3_matches(mol))


def smiles_has_cf3(smiles):
    """SMILES contains CF3 moiety

    :param smiles: SMILES string of molecule to test
    :type smiles: str
    :return: SMILES contains CF3
    :rtype: bool
    """
    mol = Chem.MolFromSmiles(smiles)
    return has_cf3(mol)


def hist_index(value, hist_bounds):
    """Return the index for a value based on the bounds

    A value below the first bound will return the index 0. A value outside the
    last bound will get the index len(hist_bounds)

    :param value: value to find a histogram index for
    :type value: float
    :param hist_bounds: list of histogram upper bounds
    :type hist_bounds: list[float]
    :return: index in the histogram defined by the bounds
    :rtype: int
    """
    index = 0
    while len(hist_bounds) > index and value > (hist_bounds[index] + np.finfo(float).eps):
        index += 1
    return index


def label_unseen(mol_index, mol, matches, label, seen):
    """Label fluroine atoms that haven't been seen yet

    :param mol_index: index of the molecule
    :type mol_index: int
    :param mol: mol that was matched
    :type mol: rdkit.Chem.rdchem.Mol
    :param matches: fluorine atom matches
    :type matches: list[list[int]]
    :param label: label to use
    :type label: str
    :param seen: set of seen atom indexes
    :type seen: set[int]
    """
    labeled = []
    for match in matches:
        if any(i in seen for i in match):
            continue

        for atom_index in match:
            if mol.GetAtomWithIdx(atom_index).GetSymbol() == "F":
                labeled.append([mol_index, mol, label, atom_index])
                seen.add(atom_index)
    return labeled


def calculate_descriptors(mols, path_length=LEFFingerprint.LEF_FPL):
    """Calculate descriptors

    The main descriptor is a LEFFingerprint for every fluorine, but we need
    identifying information for the individual fluorines.

    :param mols: list of mols
    :type mols: list[rdkit.Chem.rdchem.Mol]
    :param path_length: path length of the fingerprint
    :type path_length: int
    :return: data frame of descriptors
    :rtype: pd.DataFrame
    """
    descriptors = []
    for mol_index, mol in enumerate(mols):
        cf3_matches = get_cf3_matches(mol)
        descriptors.extend([[mol_index, mol, "CF3", cf3_match[0]] for cf3_match in cf3_matches])
        cf2_matches = get_cf2_matches(mol)
        descriptors.extend([[mol_index, mol, "CF2", cf2_match[0]] for cf2_match in cf2_matches])
        cf_matches = get_cf_matches(mol)
        descriptors.extend([[mol_index, mol, "CF", cf_match[0]] for cf_match in cf_matches])

    for row in descriptors:
        fps, _ = LEFFingerprint.generate_with_info(
            row[1], from_atoms=[[row[-1]]], path_length=path_length
        )
        assert len(fps) == 1
        row.append(fps[0])

    descriptors_df = pd.DataFrame(
        descriptors,
        columns=[
            "Index",
            constants.MOL_COLUMN,
            constants.LABEL_COLUMN,
            constants.ATOM_INDEX_COLUMN,
            constants.FP_COLUMN,
        ],
    )
    descriptors_df = descriptors_df.set_index("Index")
    return descriptors_df


def smiles_calculate_descriptors(smiles, path_length=LEFFingerprint.LEF_FPL):
    """Calculate descriptors for SMILES

    The main descriptor is a LEFFingerprint for every fluorine, but we need
    identifying information for the individual fluorines.

    :param smiles: list of SMILES strings
    :type smiles: list[str]
    :param path_length: path length of the fingerprint
    :type path_length: int
    :return: data frame of descriptors
    :rtype: pd.DataFrame
    """
    params = SmilesParserParams()
    params.removeHs = False
    mols = [Chem.MolFromSmiles(smile, params) for smile in smiles]
    return calculate_descriptors(mols, path_length=path_length)


def calculate_fingerprints(input_data, smiles_column, path_length=LEFFingerprint.LEF_FPL):
    """Calculate fingerprints for input data

    Calculate fingerprints for input data. The input data must contain
    a SMILES column and an atom index column with indexes for fluorines.

    :param input_data: input data to calculate fingerprints for
    :type input_data: pd.DataFrame
    :param smiles_column: name of the SMILES column
    :type smiles_column: str
    :return: data frame of molecules and fingerprints
    :rtype: pd.DataFrame
    """
    params = SmilesParserParams()
    params.removeHs = False
    mol_fps = []
    for _, row in input_data.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_column], params)
        if not mol:
            raise RuntimeError("Could not generate molecule from SMILES")
        if mol.GetAtomWithIdx(int(row[constants.ATOM_INDEX_COLUMN])).GetSymbol() != "F":
            raise RuntimeError(
                f'Atom at index {row[constants.ATOM_INDEX_COLUMN]} in row with ID "{row[constants.ID_COLUMN]}" is not a fluorine atom'
            )
        fps, bit_info = LEFFingerprint.generate_with_info(
            mol, from_atoms=[[row["Atom Index"]]], path_length=path_length
        )
        assert len(fps) == 1
        mol_fps.append((mol, fps[0], bit_info))
    return pd.DataFrame(
        mol_fps, columns=[constants.MOL_COLUMN, constants.FP_COLUMN, constants.FP_INFO_COLUMN]
    )


def filter_by_descriptor(input_data, column_name, calculated_name):
    """Filter by a descriptor

    Requires an input column containing the descriptor and the calculated
    descriptor. If the input descriptor of a row is empty no records are
    filtered out.

    :param input_data: input data to filter
    :type input_data: pd.DataFrame
    :param column_name: name of the column containing the descriptor
    :type column_name: str
    :param calculated_name: name of the column containing the calculated descriptor
    :type calculated_name: str
    :return: filtered data with empty values replaced
    :rtype: pd.DataFrame
    """
    comparison_column = calculated_name
    if column_name == calculated_name:
        comparison_column += " Descriptors"
    input_data = input_data[
        input_data[column_name].isna() | (input_data[column_name] == input_data[comparison_column])
    ]
    if input_data[column_name].hasnans:
        input_data = input_data.assign(**{column_name: input_data[comparison_column].values})
    input_data = input_data.drop(comparison_column, axis="columns")
    return input_data


def generate_descriptors(
    input_data,
    smiles_column=constants.SMILES_COLUMN,
    id_column=constants.ID_COLUMN,
    label_column=constants.LABEL_COLUMN,
    atom_index_column=constants.ATOM_INDEX_COLUMN,
):
    """Generate normalized descriptors

    Handles full and partial specification of fluorine motifs with labels and
    atom indexes. Will overwrite partially specified values with the fully
    specified values from descriptor calculation.

    :param input_data: input data to calculate fingerprints for
    :type input_data: pd.DataFrame
    :param smiles_column: name of the SMILES column
    :type smiles_column str
    :param id_column: name of the ID column
    :type id_column str
    :param label_column: name of the label column
    :type label_column str
    :param atom_index_column: name of the atom index column
    :type atom_index_column str
    :return: input data with descriptors
    :rtype: pd.DataFrame
    """
    smiles = input_data[smiles_column].values
    if id_column in input_data:
        smiles += " " + input_data[id_column].values  # name the smiles
    suffix = " Descriptors"
    input_data = input_data.join(
        smiles_calculate_descriptors(smiles),
        rsuffix=suffix,
    )
    if (constants.LABEL_COLUMN + suffix in input_data.columns) or (
        constants.LABEL_COLUMN != label_column and constants.LABEL_COLUMN in input_data.columns
    ):
        input_data = filter_by_descriptor(input_data, label_column, constants.LABEL_COLUMN)

    if (constants.ATOM_INDEX_COLUMN + suffix in input_data.columns) or (
        constants.ATOM_INDEX_COLUMN != atom_index_column
        and constants.ATOM_INDEX_COLUMN in input_data.columns
    ):
        input_data = filter_by_descriptor(
            input_data, atom_index_column, constants.ATOM_INDEX_COLUMN
        )
    return input_data


def string_to_list(list_string):
    """Convert a string of a list back to a list

    Does not type cast the members

    :param list_string: string of a list
    :type list_string: str
    :return: list from string
    :rtype: list[str]
    """
    list_string = list_string.replace("[", "")
    list_string = list_string.replace("]", "")
    list_string = list_string.replace("'", "")
    return [e for e in re.split(r",?\s+", list_string) if e]


def have_common_neighbor(atoms):
    """Checks whether the atoms are connected to one common atom

    Iterative set intersection between all neighbors of all atoms. Runs in O(n)
    where n is the number of neighbors of all atoms

    :param atoms: list of atoms
    :type atoms: list[rdkit.Chem.rdchem.Atom]
    :return: atoms have common neighbor
    :rtype: bool
    """
    common_neighbors = None
    for atom in atoms:
        bonds = atom.GetBonds()
        neighbors = set()
        for bond in bonds:
            # putting both begin and end in the set because checking which one
            # is the current atom is not worth it
            neighbors.add(bond.GetBeginAtomIdx())
            neighbors.add(bond.GetEndAtomIdx())
        if common_neighbors is None:
            common_neighbors = neighbors
        else:
            common_neighbors = common_neighbors.intersection(neighbors)
    return len(common_neighbors) > 0


def indexes_differ(mol, other_mol):
    """Check if atom indexes differ"""
    if mol.GetNumAtoms() != other_mol.GetNumAtoms():
        return True

    return any(
        atom.GetSymbol() != other_atom.GetSymbol()
        for atom, other_atom in zip(mol.GetAtoms(), other_mol.GetAtoms())
    )


def map_atom_indexes(mol, reference):
    """Get an atom index mapping from mol to reference

    :param mol: mol to map from
    :type mol: rdkit.Chem.rdchem.Mol
    :param reference: mol to map to
    :type reference: rdkit.Chem.rdchem.Mol
    :rtype: map from mol indexes to reference indexes
    :return: dict[int, int]
    """
    mcs = rdFMCS.FindMCS([reference, mol], bondCompare=rdFMCS.BondCompare.CompareAny)
    match_pattern = Chem.MolFromSmarts(mcs.smartsString)
    reference_match = reference.GetSubstructMatch(match_pattern)
    current_match = mol.GetSubstructMatch(match_pattern)
    match_map = {}
    for reference_index, current_index in zip(reference_match, current_match):
        match_map[current_index] = reference_index
    return match_map


def get_ensemble_stereo(ensemble):
    """Get stereo information for an ensemble

    Makes the assumption that atom indexes between ensemble memebers do not
    change

    :param ensemble: ensemble of molecules
    :type ensemble: list[rdkit.Chem.rdchem.Mol]
    :return: chiral tags for ensemble, stereo indexes for ensemble, configurations for ensemble
    :rtype: list[list[rdkit.Chem.rdchem.ChiralType]], tuple[int], list[tuple[rdkit.Chem.rdchem.ChiralType]]
    """
    stereo_indexes = set()
    stereo_groups = []
    reference_mol = ensemble[0]
    for mol in ensemble:
        match_map = None
        if indexes_differ(mol, reference_mol):
            match_map = map_atom_indexes(mol, reference_mol)
        stereo_group = []
        for atom in mol.GetAtoms():
            stereo_group.append(atom.GetChiralTag())
            if atom.GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                continue
            if match_map is None:
                stereo_indexes.add(atom.GetIdx())
            else:
                stereo_indexes.add(match_map[atom.GetIdx()])
        stereo_groups.append(stereo_group)
    stereo_indexes = tuple(sorted(stereo_indexes))

    configurations = []
    for stereo_group in stereo_groups:
        configuration = []
        for stereo_index in stereo_indexes:
            configuration.append(stereo_group[stereo_index])
        configurations.append(tuple(configuration))
    return stereo_groups, stereo_indexes, configurations


def bit_difference(fingerprint, bits):
    """Get number of different bits between the fingerprint and the bits

    :param fingerprint: fingerprint with different bits
    :type fingerprint: rdkit.DataStructs.cDataStructs.LongSparseIntVect
    :param bits: set of bits
    :type bits: set[int]
    :return: number of different bits in fingerprint
    :rtype: int
    """
    return len(set(fingerprint.GetNonzeroElements().keys()).difference(bits))


def get_bit_set(fingerprints):
    """Get the set of present bits in fingerprints

    :param fingerptins: fingerprints to get bits for
    :type fingerprint: list[rdkit.DataStructs.cDataStructs.LongSparseIntVect]
    :return: set of bits present in the fingerprints
    :rtype: set[int]
    """
    sum_fingerprint = np.sum([fingerprint.ToList() for fingerprint in fingerprints], axis=0)
    return {bit for bit, value in enumerate(sum_fingerprint) if value > 0}
