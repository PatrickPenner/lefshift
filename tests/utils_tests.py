"""Test utils"""
import unittest

import pandas as pd
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

from lefshift import utils
from lefshift.fp import LEFFingerprint


class UtilsTests(unittest.TestCase):
    """Test utils"""

    def test_smiles_nof_unique_fingerprints(self):
        """Test has nof unique fingerprints"""
        smiles = "Cc1nc(C(F)(F)F)ccc1N Z1021421710"
        self.assertEqual(utils.smiles_nof_unique_fingerprints(smiles), 1)
        smiles = "OCc1cccc(F)c1F CHEMBL1232182"
        self.assertEqual(utils.smiles_nof_unique_fingerprints(smiles), 2)

    def test_highest_dice_similarity(self):
        """Test finding highest dice similarity"""
        smiles = "Cc1nc(C(F)(F)F)ccc1N Z1021421710"
        other_smiles = [
            "OCc1cccc(F)c1F CHEMBL1232182",
            "Cc1nc(C(F)(F)F)ccc1N Z1021421710",
        ]
        mol = Chem.MolFromSmiles(smiles)
        mols = [Chem.MolFromSmiles(smiles) for smiles in other_smiles]
        fingerprint = LEFFingerprint.generate_with_info(mol)[0][0]
        fingerprints = [LEFFingerprint.generate_with_info(mol)[0][0] for mol in mols]
        similarity, index = utils.highest_dice_similarity(fingerprint, fingerprints)
        # equal 1.0
        self.assertAlmostEqual(similarity, 1.0)
        self.assertEqual(index, 1)

    def test_sorted_dice_neighbors(self):
        """Test extracting sorted neighbors by dice similarity"""
        smiles = "Cc1nc(C(F)(F)F)ccc1N Z1021421710"
        other_smiles = [
            "OCc1cccc(F)c1F CHEMBL1232182",
            "Cc1nc(C(F)(F)F)ccc1N Z1021421710",
        ]
        mol = Chem.MolFromSmiles(smiles)
        mols = [Chem.MolFromSmiles(smiles) for smiles in other_smiles]
        fingerprint = LEFFingerprint.generate_with_info(mol)[0][0]
        fingerprints = [LEFFingerprint.generate_with_info(mol)[0][0] for mol in mols]
        neighbors = utils.sorted_dice_neighbors(fingerprint, fingerprints)
        self.assertEqual([n[1] for n in neighbors], [1, 0])

    def test_has_cf(self):
        """Test SMILES has a CF"""
        smiles = "N#Cc1ccc(N2CCCOCC2)c(F)c1 Z1002244320"
        self.assertTrue(utils.smiles_has_cf(smiles))
        smiles = "OCc1cccc(F)c1F CHEMBL1232182"
        self.assertTrue(utils.smiles_has_cf(smiles))
        smiles = "Cc1nc(C(F)(F)F)ccc1N Z1021421710"
        self.assertFalse(utils.smiles_has_cf(smiles))

    def test_has_cf2(self):
        """Test SMILES has a CF2"""
        smiles = "O=C(c1sccc1OC(F)F)N1CCCC1 Z1127699252"
        self.assertTrue(utils.smiles_has_cf2(smiles))
        smiles = "OCc1cccc(F)c1F CHEMBL1232182"
        self.assertFalse(utils.smiles_has_cf2(smiles))
        smiles = "Cc1nc(C(F)(F)F)ccc1N Z1021421710"
        self.assertFalse(utils.smiles_has_cf2(smiles))

    def test_has_cf3(self):
        """Test SMILES has a CF3"""
        smiles = "O=C(c1sccc1OC(F)F)N1CCCC1 Z1127699252"
        self.assertFalse(utils.smiles_has_cf3(smiles))
        smiles = "Cc1nc(C(F)(F)F)ccc1N Z1021421710"
        self.assertTrue(utils.smiles_has_cf3(smiles))

    def test_hist_index(self):
        """Test hist indexng function"""
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.assertEqual(utils.hist_index(0.65, bins), 2)
        self.assertEqual(utils.hist_index(0.45, bins), 0)
        self.assertEqual(utils.hist_index(1.0, bins), 5)
        self.assertEqual(utils.hist_index(1.1, bins), 6)

    def test_calculate_descriptors(self):
        """Test descriptor calculation"""
        smiles = [
            "OCc1cccc(F)c1F CHEMBL1232182",
            "O=C(c1sccc1OC(F)F)N1CCCC1 Z1127699252",
            "Cc1nc(C(F)(F)F)ccc1N Z1021421710",
        ]
        descriptors = utils.smiles_calculate_descriptors(smiles)
        self.assertEqual(len(descriptors), 4)

        # ensure descriptor calculation does not strip hydrogens
        smiles = ["[H]C([H])(F)C(=O)O CHEMBL509273"]
        descriptors = utils.smiles_calculate_descriptors(smiles)
        # 2 hydrogens and one carbon are in front of the fluorine
        self.assertEqual(descriptors["Atom Index"].values[0], 3)

    def test_generate_descriptors(self):
        """Test descriptor generation"""
        input_data = pd.read_csv("tests/data/test_data.csv")
        input_data_descriptors = utils.generate_descriptors(input_data)
        self.assertEqual(len(input_data), len(input_data_descriptors))
        self.assertFalse(input_data_descriptors["Label"].hasnans)
        self.assertFalse(input_data_descriptors["Atom Index"].hasnans)

        input_data = pd.read_csv("tests/data/test_data_custom_columns.csv")
        input_data_descriptors = utils.generate_descriptors(input_data, label_column="My Label")
        self.assertEqual(len(input_data), len(input_data_descriptors))
        self.assertFalse(input_data_descriptors["My Label"].hasnans)
        self.assertFalse(input_data_descriptors["Atom Index"].hasnans)

    def test_string_to_list(self):
        """Test string to list conversion"""
        test_list = ["a", "b", "c"]
        list_string = str(test_list)
        self.assertEqual(utils.string_to_list(list_string), test_list)

        test_list = ["1", "2", "3"]
        list_string = "['1', '2',\n'3']"
        self.assertEqual(utils.string_to_list(list_string), test_list)

        test_list = [1, 2, 3]
        list_string = str(test_list)
        # does not type cast members
        self.assertNotEqual(utils.string_to_list(list_string), test_list)

    def test_have_common_neighbor(self):
        """Test common neighbor search"""
        mol = Chem.MolFromSmiles("O=C(c1sccc1OC(F)F)N1CCCC1 Z1127699252")
        fluorine_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "F"]
        self.assertTrue(utils.have_common_neighbor(fluorine_atoms))

        mol = Chem.MolFromSmiles("OCc1cccc(F)c1F CHEMBL1232182")
        fluorine_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "F"]
        self.assertFalse(utils.have_common_neighbor(fluorine_atoms))

        mol = Chem.MolFromSmiles("Cc1nc(C(F)(F)F)ccc1N Z1021421710")
        fluorine_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "F"]
        self.assertTrue(utils.have_common_neighbor(fluorine_atoms))

    def test_get_ensemble_stereo(self):
        """Test getting ensemble stereo information"""
        mol = Chem.MolFromSmiles("CC(C)C1CCC(C)CC1O CHEMBL470670")
        ensemble = list(EnumerateStereoisomers(mol))
        # enumerated 3 unassigned stereo centers
        self.assertEqual(len(ensemble), 8)
        stereo_groups, stereo_indexes, configurations = utils.get_ensemble_stereo(ensemble)
        # stereo groups are list of chiral tags for each atom of each mol
        self.assertEqual(len(stereo_groups), 8)
        self.assertEqual(len(stereo_groups[0]), mol.GetNumAtoms())
        # stereo indexes are stereo centers that change in the ensemble
        self.assertEqual(len(stereo_indexes), 3)
        # configurations are chiral tags for each of the changing stereo centers
        self.assertEqual(len(configurations[0]), 3)
        # in this case all configurations are unique
        self.assertEqual(len(set(configurations)), 8)

        # if we fix all but on stereocenter only one should vary in the ensemble
        mol = Chem.MolFromSmiles("CC(C)[C@@H]1CC[C@@H](C)CC1O CHEMBL470670")
        ensemble = list(EnumerateStereoisomers(mol))
        stereo_groups, stereo_indexes, configurations = utils.get_ensemble_stereo(ensemble)
        # two unique configurations at one stereo center
        self.assertEqual(len(set(configurations)), 2)

    def test_map_atom_indexes(self):
        """Test mapping of atom indices"""
        mol1 = Chem.MolFromSmiles("C=CC1CO1 CHEMBL1299388")
        mol2 = Chem.MolFromSmiles("C=CC1OC1 CHEMBL1299388")
        self.assertTrue(utils.indexes_differ(mol1, mol2))
        match_map = utils.map_atom_indexes(mol1, mol2)
        for atom in mol1.GetAtoms():
            atom2_index = match_map[atom.GetIdx()]
            self.assertEqual(atom.GetSymbol(), mol2.GetAtomWithIdx(atom2_index).GetSymbol())

    def test_indexes_differ(self):
        """Test indexes differ check"""
        mol1 = Chem.MolFromSmiles("C=CC1CO1 CHEMBL1299388")
        mol2 = Chem.MolFromSmiles("C=CC1OC1 CHEMBL1299388")
        self.assertTrue(utils.indexes_differ(mol1, mol2))

    def test_bit_difference(self):
        """Test bit difference"""
        mol = Chem.MolFromSmiles("S(=O)(=O)(N(CC)C)c1cncc(F)c1 Z1188305542")
        fingerprint = LEFFingerprint.generate_with_info(mol)[0][0]
        bits = fingerprint.GetNonzeroElements().keys()
        other_mol = Chem.MolFromSmiles("FC(F)CNC(=O)c1nc2NC=Cc2cc1 Z1625511334")
        other_fingerprint = LEFFingerprint.generate_with_info(other_mol)[0][0]
        other_bits = other_fingerprint.GetNonzeroElements().keys()
        # all bits are different
        self.assertEqual(utils.bit_difference(other_fingerprint, bits), len(other_bits))
        # all bits are the same
        self.assertEqual(utils.bit_difference(other_fingerprint, other_bits), 0)

    def test_get_bit_set(self):
        """Test generating bit sets"""
        mol = Chem.MolFromSmiles("S(=O)(=O)(N(CC)C)c1cncc(F)c1 Z1188305542")
        fingerprint = LEFFingerprint.generate_with_info(mol)[0][0]
        expected_set = {
            6653,
            7709,
            5349,
            4774,
            7270,
            8070,
            1291,
            8030,
            2194,
            7379,
            2325,
            3031,
            2235,
            4765,
            990,
            7103,
        }
        self.assertEqual(utils.get_bit_set([fingerprint]), expected_set)
        other_mol = Chem.MolFromSmiles("FC(F)CNC(=O)c1nc2NC=Cc2cc1 Z1625511334")
        other_fingerprint = LEFFingerprint.generate_with_info(other_mol)[0][0]
        expected_set.update({7139, 6412, 1868, 3886, 7537, 5812, 8052, 4314, 6556})
        self.assertEqual(utils.get_bit_set([fingerprint, other_fingerprint]), expected_set)
