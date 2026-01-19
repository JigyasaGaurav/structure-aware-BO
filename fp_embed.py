import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mol_fingerprints import build_ligand_fingerprint_dict

class FingerprintEmbedder:
    def __init__(self, smiles_path, nBits, target_column=None):
        """
        Initialize the embedder with a SMILES file, fingerprint size, and optional target column.
        """
        self.nBits = nBits
        self.ligand2fp = build_ligand_fingerprint_dict(smiles_path, nBits)
        self.feature_columns = None
        self.target_column = target_column
        self.target_encoder = None

    def encode_features(self, df):
        """
        Encode ligand pair features into fingerprint vectors and concatenate with n_1, n_2 columns.
        Args:
            df (pd.DataFrame): Input dataframe with 'Ligand 1', 'Ligand 2', 'n_1', 'n_2' columns.
        Returns:
            pd.DataFrame: DataFrame with encoded features.
        """
        df = df.copy()
        fp_cols_1 = [f"Ligand1_fp_{i}" for i in range(self.nBits)]
        fp_cols_2 = [f"Ligand2_fp_{i}" for i in range(self.nBits)]
        fp1 = df['Ligand 1'].apply(lambda x: self.ligand2fp.get(str(x), np.zeros(self.nBits, dtype=int)))
        fp2 = df['Ligand 2'].apply(lambda x: self.ligand2fp.get(str(x), np.zeros(self.nBits, dtype=int)))
        fp1_df = pd.DataFrame(fp1.tolist(), columns=fp_cols_1, index=df.index)
        fp2_df = pd.DataFrame(fp2.tolist(), columns=fp_cols_2, index=df.index)
        for col in ['n_1', 'n_2']:
            df[col] = df.get(col, 0.0)
        df_final = pd.concat([fp1_df, fp2_df, df[['n_1', 'n_2']]], axis=1)
        self.feature_columns = list(df_final.columns)
        return df_final

    def encode_target(self, y, target_column=None):
        """
        Encode the target column using LabelEncoder if it is categorical.
        Args:
            y (pd.DataFrame): DataFrame containing the target column.
            target_column (str): Name of the target column.
        Returns:
            pd.DataFrame: DataFrame with encoded target column.
        """
        if target_column is not None:
            self.target_column = target_column
        if self.target_column is None:
            raise ValueError("target_column must be specified either in constructor or in encode_target call.")
        y = y.copy()
        if y[self.target_column].dtype == object or str(y[self.target_column].dtype).startswith('category'):
            self.target_encoder = LabelEncoder()
            y[self.target_column] = self.target_encoder.fit_transform(y[self.target_column].astype(str))
        return y
