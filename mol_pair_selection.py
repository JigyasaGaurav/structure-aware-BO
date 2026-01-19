import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn_extra.cluster import KMedoids
from config_loader import SMILES_FILE, FINGERPRINT_NBITS, NUM_CLUSTERS


class MolPairSelector:
    '''
    Selects diverse molecular pairs from a list of SMILES strings using Tanimoto similarity and K-medoids clustering.
    Args:
        Path to a file containing SMILES strings.
    Returns:
        A list of selected molecular pairs.
    '''
    def __init__(self, smiles_file: SMILES_FILE):
        self.smiles_file = smiles_file
        self.nBits = FINGERPRINT_NBITS
        self.mols = self.read_smiles()
        self.fps = self.get_fps()
        self.d_lig = self.ligand_distance_matrix()
        self.pairs, self.D_pair = self.pair_distance_matrix()

    def read_smiles(self) -> list[Chem.Mol]:
        '''
        Reads SMILES strings from file and converts them to RDKit Mol objects.
        Returns:
            A list of Mol objects.
        '''
        mols = []
        with open(self.smiles_file, 'r') as f:
            for line in f:
                smiles = line.strip().split(': ')[1]
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols.append(mol)
                else:
                    print(f"Invalid smiles: {line.strip()}")
        return mols

    def get_fps(self) -> list[np.ndarray]:
        '''
        Generates Morgan fingerprints for each molecule.
        Returns:
            A list of Morgan fingerprint bit vectors.
        '''
        return [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=self.nBits) for m in self.mols]

    def tanimoto(self, i, j) -> float:
        '''
        Computes the Tanimoto similarity between two fingerprints.
        Args:
            i (int): Index of the first fingerprint.
            j (int): Index of the second fingerprint.
        Returns:
            Tanimoto similarity between the two fingerprints.
        '''
        return DataStructs.TanimotoSimilarity(self.fps[i], self.fps[j])

    def ligand_distance_matrix(self) -> np.ndarray:
        '''
        Computes the ligand distance matrix based on Tanimoto similarity.
        Returns:
            A square matrix where each element (i, j) is 1 - Tanimoto similarity between fingerprints i and j.
        '''
        n = len(self.fps)
        return np.array([[1.0 - self.tanimoto(i, j) for j in range(n)] for i in range(n)])

    def pair_distance_matrix(self) -> tuple[list[tuple[int, int]], np.ndarray]:
        '''
        Computes the pair distance matrix for all unique pairs of molecules.
        Returns:
            A tuple containing:
            - A list of tuples representing pairs of molecule indices.
            - A square matrix where each element (a, b) is the distance between the pair of molecules.
        '''
        pairs = [(i, j) for i in range(len(self.d_lig)) for j in range(i + 1, len(self.d_lig))]
        P = len(pairs)
        D_pair = np.zeros((P, P))
        for a, (i, j) in enumerate(pairs):
            for b, (p, q) in enumerate(pairs[a:], start=a):
                d1 = self.d_lig[i, p] + self.d_lig[j, q]
                d2 = self.d_lig[i, q] + self.d_lig[j, p]
                D_pair[a, b] = D_pair[b, a] = 0.5 * min(d1, d2)
        return pairs, D_pair

    def choose_mol_pairs(self, K=NUM_CLUSTERS) -> list[tuple[str, str]]:
        '''
        Selects K diverse molecular pairs using K-medoids clustering.
        Args:
            K (int): Number of pairs to select.
        Returns:
            A list of selected molecular pairs represented as tuples of SMILES strings.
        '''
        med = KMedoids(n_clusters=K, metric='precomputed', init='k-medoids++', random_state=42).fit(self.D_pair)
        chosen_pairs = [self.pairs[idx] for idx in med.medoid_indices_]
        chosen_smiles_pairs = [
            (Chem.MolToSmiles(self.mols[i]), Chem.MolToSmiles(self.mols[j]))
            for i, j in chosen_pairs
        ]
        return chosen_smiles_pairs

if __name__ == "__main__":
    selector = MolPairSelector()
    chosen_smiles_pairs = selector.choose_mol_pairs()
    print(chosen_smiles_pairs)
