import os
import yaml

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_directories_exist(config):
    """Create directories for all path configurations if they don't exist"""
    paths_to_check = []
    
    for path_key, path_value in config['paths'].items():
        if path_key in ['smiles_file', 'data_path', 'new_data_path', 'train_path', 
                        'test_path', 'model_path', 'results_path', 'analysis_output', 
                        'plot_data_path', 'plot_path']:
            # Add null check here
            if path_value is not None:
                dir_path = os.path.dirname(path_value)
                if dir_path and dir_path not in paths_to_check:
                    paths_to_check.append(dir_path)

    for dir_path in paths_to_check:
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"{dir_path} created.")

_config = load_config()

# Ensure all required directories exist
ensure_directories_exist(_config)

# Paths
SMILES_FILE = _config['paths']['smiles_file']
DATA_PATH = _config['paths']['data_path']
NEW_DATA_PATH = _config['paths']['new_data_path']
TRAIN_PATH = _config['paths']['train_path']
TEST_PATH = _config['paths']['test_path']
MODEL_PATH = _config['paths']['model_path']
RESULTS_PATH = _config['paths']['results_path']
ANALYSIS_OUTPUT = _config['paths']['analysis_output']
PLOT_DATA_PATH = _config['paths']['plot_data_path']
PLOT_PATH = _config['paths']['plot_path']

# Model parameters
FINGERPRINT_NBITS = _config['model']['fingerprint_nBits']
TRAINING_ITERATIONS = _config['model']['training_iterations']
NUM_CLUSTERS = _config['model']['num_clusters']

# Data processing parameters
TEST_SIZE = _config['data_processing']['test_size']
SEED = _config['data_processing'].get('seed', 42)
TARGET_COLUMNS = _config['data_processing']['target_columns']

# GP Model parameters (for from_params)
GP_PARAMS = _config.get('gp_params', {'noise_type': 'fixed', 'fixed_noise_value': 0.05})

# Helper function to get the full config if needed
def get_full_config():
    """Get the full configuration dictionary"""
    return _config