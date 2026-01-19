import pandas as pd
import random
import os

def modify_ligand_data(input_file, output_file=None):
    df = pd.read_csv(input_file)
    mask = df['Ligand 2'] == '-'
    for idx in df[mask].index:
        df.loc[idx, 'Ligand 2'] = df.loc[idx, 'Ligand 1']
        n1 = 12 # random.randint(1, 23)
        n2 = 24 - n1
        df.loc[idx, 'n_1'] = n1
        df.loc[idx, 'n_2'] = n2
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        extension = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_modified{extension}"
    df.to_csv(output_file, index=False)    
    return df

if __name__ == "__main__":
    input_file = "data/ligand_data.csv"
    modified_df = modify_ligand_data(input_file, "data/ligand_data_modified.csv")