"""Test LEF fingerprint"""
import unittest

from rdkit import Chem

from lefshift.fp import LEFFingerprint


class LEFFingerprintTests(unittest.TestCase):
    """Test LEF Fingerprint"""

    def test_generate(self):
        """Test generation of LEF fingerprints"""
        smiles = "S(=O)(=O)(N(CC)C)c1cncc(F)c1 Z1188305542"
        mol = Chem.MolFromSmiles(smiles)
        fingerprints = LEFFingerprint.generate(mol, 7)
        self.assertEqual(
            [fingerprint.GetNonzeroElements() for fingerprint in fingerprints],
            [
                {
                    990: 1,
                    1291: 2,
                    2194: 1,
                    2235: 1,
                    2325: 1,
                    3031: 1,
                    4765: 1,
                    4774: 1,
                    5349: 1,
                    6653: 1,
                    7103: 2,
                    7270: 1,
                    7379: 1,
                    7709: 1,
                    8030: 1,
                    8070: 1,
                }
            ],
        )

        smiles = "FC(F)CNC(=O)c1nc2NC=Cc2cc1 Z1625511334"
        mol = Chem.MolFromSmiles(smiles)
        fingerprints = LEFFingerprint.generate(mol, 7)
        self.assertEqual(
            [fingerprint.GetNonzeroElements() for fingerprint in fingerprints],
            [
                {1868: 1, 3886: 1, 4314: 1, 5812: 1, 6412: 1, 6556: 1, 7139: 1, 7537: 1, 8052: 1},
                {1868: 1, 3886: 1, 4314: 1, 5812: 1, 6412: 1, 6556: 1, 7139: 1, 7537: 1, 8052: 1},
            ],
        )

        smiles = "FC(F)(F)c1nc(nc(c1)C)N2C=NC=C2 Z1665678644"
        mol = Chem.MolFromSmiles(smiles)
        fingerprints = LEFFingerprint.generate(mol, 7)
        self.assertEqual(
            [fingerprint.GetNonzeroElements() for fingerprint in fingerprints],
            [
                {
                    2181: 1,
                    2451: 1,
                    3338: 1,
                    3430: 1,
                    3628: 1,
                    4208: 1,
                    4890: 2,
                    5133: 1,
                    5138: 1,
                    5375: 1,
                    5740: 1,
                    6024: 1,
                    6605: 2,
                    7005: 1,
                },
                {
                    2181: 1,
                    2451: 1,
                    3338: 1,
                    3430: 1,
                    3628: 1,
                    4208: 1,
                    4890: 2,
                    5133: 1,
                    5138: 1,
                    5375: 1,
                    5740: 1,
                    6024: 1,
                    6605: 2,
                    7005: 1,
                },
                {
                    2181: 1,
                    2451: 1,
                    3338: 1,
                    3430: 1,
                    3628: 1,
                    4208: 1,
                    4890: 2,
                    5133: 1,
                    5138: 1,
                    5375: 1,
                    5740: 1,
                    6024: 1,
                    6605: 2,
                    7005: 1,
                },
            ],
        )

    def test_generate_with_info(self):
        """Test generation of LEF fingerprints with bit information"""
        smiles = "S(=O)(=O)(N(CC)C)c1cncc(F)c1 Z1188305542"
        mol = Chem.MolFromSmiles(smiles)
        fingerprints, bit_info = LEFFingerprint.generate_with_info(mol)
        self.assertEqual(
            [fingerprint.GetNonzeroElements() for fingerprint in fingerprints],
            [
                {
                    990: 1,
                    1291: 2,
                    2194: 1,
                    2235: 1,
                    2325: 1,
                    3031: 1,
                    4765: 1,
                    4774: 1,
                    5349: 1,
                    6653: 1,
                    7103: 2,
                    7270: 1,
                    7379: 1,
                    7709: 1,
                    8030: 1,
                    8070: 1,
                }
            ],
        )
        self.assertEqual(len(bit_info), len(fingerprints[0].GetNonzeroElements()))
        self.assertEqual(bit_info.keys(), fingerprints[0].GetNonzeroElements().keys())
