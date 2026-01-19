import pandas as pd
import torch
from data_processing import process_data
from gp_model import GPModel
from config_loader import MODEL_PATH, TARGET_COLUMNS, TEST_PATH, RESULTS_PATH, SMILES_FILE, FINGERPRINT_NBITS
from fp_embed import FingerprintEmbedder

def train_gp(train_X, train_y, model_path=MODEL_PATH, target_column=TARGET_COLUMNS[0]):
    # Initialize embedder
    embedder = FingerprintEmbedder(smiles_path=SMILES_FILE, nBits=FINGERPRINT_NBITS)
    
    # Encode features
    X_enc = embedder.encode_features(train_X)
    train_x_tensor = torch.tensor(X_enc.values, dtype=torch.double)
    
    # Encode target
    if isinstance(train_y, pd.DataFrame):
         train_y_vals = train_y[target_column].values
    else:
         train_y_vals = train_y
         
    train_y_tensor = torch.tensor(train_y_vals, dtype=torch.double)

    # Init GP with nBits
    gp = GPModel(nBits=FINGERPRINT_NBITS)
    
    # Fit
    gp.fit(train_x_tensor, train_y_tensor)
    
    # Save
    gp.save_model(model_path)

def predict_and_save(
        test_csv_path=TEST_PATH, 
        model_path=MODEL_PATH, 
        target_column=TARGET_COLUMNS[0], 
        save_path=RESULTS_PATH
):
    test_df = pd.read_csv(test_csv_path)
    
    embedder = FingerprintEmbedder(smiles_path=SMILES_FILE, nBits=FINGERPRINT_NBITS)
    X_enc = embedder.encode_features(test_df)
    test_x_tensor = torch.tensor(X_enc.values, dtype=torch.double)

    gp = GPModel(nBits=FINGERPRINT_NBITS)
    
    # We need shapes to load model if not fitting.
    input_dim = test_x_tensor.shape[-1]
    train_x_shape = (1, input_dim) 
    train_y_shape = (1,) 
    
    gp.load_model(model_path, train_x_shape=train_x_shape, train_y_shape=train_y_shape)
    
    preds = gp.predict(test_x_tensor)
    test_df[target_column + '_pred'] = preds
    test_df.to_csv(save_path, index=False)
    return test_df

def main():
    train_X, train_y, test_X, test_y = process_data()
    train_gp(train_X, train_y)
    abs_pred_df = predict_and_save()
    print("Sample predictions:\n", abs_pred_df.filter(regex='Ligand|n_|pred').head())
    
if __name__ == "__main__":
    main()