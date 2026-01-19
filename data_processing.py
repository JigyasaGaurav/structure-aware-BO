import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config_loader import SEED, DATA_PATH, TRAIN_PATH, TEST_PATH, TEST_SIZE, TARGET_COLUMNS, NEW_DATA_PATH

def clean_data(data):
    for col in ['Ligand 1', 'Ligand 2']:
        data[col] = data[col].replace('-', 'Z').fillna('Z')
    for col in ['n_1', 'n_2']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
    return data

def update_and_split_data(
    data_path=DATA_PATH,
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    new_data_path=NEW_DATA_PATH,
    test_size=TEST_SIZE
):    
    data = pd.read_csv(data_path)
    data = clean_data(data)

    # If train/test don't exist, split all data
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        train, test = train_test_split(data, test_size=test_size, random_state=SEED)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        print("Initial split done.")
        return

    # If new data is provided, merge it and update ligand_data.csv
    if new_data_path is not None:
        new_data = pd.read_csv(new_data_path)
        new_data = clean_data(new_data)
        # Remove duplicates based on all columns
        data = pd.concat([data, new_data], ignore_index=True).drop_duplicates()
        data.to_csv(data_path, index=False)
        # Only split the new data
        data_to_split = new_data
        # Split and append to train/test
        train_new, test_new = train_test_split(data_to_split, test_size=test_size, random_state=SEED)
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        train = pd.concat([train, train_new], ignore_index=True).drop_duplicates()
        test = pd.concat([test, test_new], ignore_index=True).drop_duplicates()
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        print("New data split and appended.")
        return

    print("Train/test already exist and no new data provided. No data splitting required.")

def process_data(
        data_path=DATA_PATH, 
        test_size=TEST_SIZE, 
        train_path=TRAIN_PATH, 
        test_path=TEST_PATH, 
        target_columns=TARGET_COLUMNS,
        new_data_path=NEW_DATA_PATH
):
    update_and_split_data(data_path, train_path, test_path, new_data_path, test_size)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    for col in ['n_1', 'n_2']:
        train[col] = train[col] / 24.0
        test[col] = test[col] / 24.0

    if 'wavelength' in target_columns:
        train_range = train['wavelength'].max() - train['wavelength'].min()
        train['wavelength'] = (train['wavelength'] - train['wavelength'].min()) / train_range
        test['wavelength'] = (test['wavelength'] - train['wavelength'].min()) / train_range

    train_X = train[['Ligand 1', 'n_1', 'Ligand 2', 'n_2']]
    train_y = train[target_columns]
    test_X = test[['Ligand 1', 'n_1', 'Ligand 2', 'n_2']]
    test_y = test[target_columns]

    return train_X, train_y, test_X, test_y