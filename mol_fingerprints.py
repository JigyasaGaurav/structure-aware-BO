import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from config_loader import SMILES_FILE, FINGERPRINT_NBITS

def load_ligand_smiles(smiles_path=SMILES_FILE):
    ligand2smiles = {}
    with open(smiles_path, 'r') as f:
        for line in f:
            if ':' in line:
                code, smiles = line.strip().split(': ')
                ligand2smiles[code] = smiles
    return ligand2smiles

def compute_fingerprint(smiles, nBits=FINGERPRINT_NBITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def build_ligand_fingerprint_dict(smiles_path=SMILES_FILE, nBits=FINGERPRINT_NBITS):
    ligand2smiles = load_ligand_smiles(smiles_path)
    ligand2fp = {lig: compute_fingerprint(smiles, nBits) for lig, smiles in ligand2smiles.items()}
    return ligand2fp