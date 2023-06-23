"""LEF Fingerprint"""
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


class LEFFingerprint:
    """
    LEF fingerprints are topological torsion based fingerpints that start at
    a fluorine motif. CF and CF3 moieties are expected. CF3 moieties are
    converted to CF. Fingerprints of all lengths starting at a lenght of 2
    and extending up to path_length are created and the occurrences of bit
    counted to generate a count fingerprint.

    Anna Vulpetti, Ulrich Hommel, Gregory Landrum, Richard Lewis, and
    Claudio Dalvit Journal of the American Chemical Society 2009 131 (36),
    12949-12959 https://doi.org/10.1021/ja905207t
    """

    LEF_FPL = 7
    GENERATORS = {
        8192: [
            rdFingerprintGenerator.GetTopologicalTorsionGenerator(torsionAtomCount=i, fpSize=8192)
            for i in range(2, LEF_FPL + 1)
        ]
    }

    @staticmethod
    def generate(mol, path_length=LEF_FPL, from_atoms=None, n_bits=8192):
        """Generate LEF fingerprints

        :param mol: molecule to generate fingerprint for
        :type mol: rdkit.Chem.rdchem.Mol
        :param path_length: maximum path length of the fingerprint
        :type path_length: int
        :param from_atoms: what atoms to calculate paths from
        :type from_atoms: list[list[int]]
        :param n_bits: number of bits for the fingerprint
        :type n_bits: int
        :return: LEF fingerprint
        :rtype: list[rdkit.DataStructs.cDataStructs.IntSparseIntVect]
        """
        return LEFFingerprint.generate_with_info(mol, path_length, from_atoms, n_bits)[0]

    @staticmethod
    def generate_with_info(mol, path_length=LEF_FPL, from_atoms=None, n_bits=8192):
        """Generate LEF fingerprints with bit information

        :param mol: molecule to generate fingerprint for
        :type mol: rdkit.Chem.rdchem.Mol
        :param path_length: maximum path length of the fingerprint
        :type path_length: int
        :param from_atoms: what atoms to calculate paths from
        :type from_atoms: list[list[int]]
        :param n_bits: number of bits for the fingerprint
        :type n_bits: int
        :return: LEF fingerprint and bit information
        :rtype: list[rdkit.DataStructs.cDataStructs.IntSparseIntVect], dict[int, list[int]]
        """
        if path_length < 2:
            raise RuntimeError(f"Invalid path length {path_length} passed to LEF fingerprint")
        if not from_atoms:
            f_query = Chem.MolFromSmarts("F")
            from_atoms = mol.GetSubstructMatches(f_query)

        if n_bits not in LEFFingerprint.GENERATORS:
            LEFFingerprint.GENERATORS[n_bits] = []

        if path_length > len(LEFFingerprint.GENERATORS[n_bits]) + 1:
            for i in range(len(LEFFingerprint.GENERATORS[n_bits]) + 2, path_length + 1):
                generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
                    torsionAtomCount=i
                )
                LEFFingerprint.GENERATORS[n_bits].append(generator)

        bit_information = {}
        fingerprints = []
        for atom_position in from_atoms:
            fingerprint = None
            for sub_generator in LEFFingerprint.GENERATORS[n_bits][:path_length]:
                additional_output = rdFingerprintGenerator.AdditionalOutput()
                additional_output.AllocateBitPaths()
                sub_fingerprint = sub_generator.GetCountFingerprint(
                    mol, fromAtoms=atom_position, additionalOutput=additional_output
                )
                bit_information.update(additional_output.GetBitPaths())
                if fingerprint is None:
                    fingerprint = sub_fingerprint
                else:
                    for bit, value in sub_fingerprint.GetNonzeroElements().items():
                        fingerprint[bit] += value
            fingerprints.append(fingerprint)
        return fingerprints, bit_information
