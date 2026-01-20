# Structure-aware Bayesian Optimization

## Project Description

This repository implements a **structure-aware Bayesian Optimization (BO)** framework designed to accelerate the discovery of porous molecular materials, specifically porous coordination cages. Designing these materials involves navigating a vast compositional space of organic ligands and mixing ratios, where physical evaluation is resource-intensive. This accompanies our paper "Bayesian Optimization Guided Lab-in-the-loop Design of Porous Molecular Materials".

We formulate this material design challenge as a black-box optimization problem over a mixture space of structured objects. We propose a structure-aware Bayesian Optimization framework by developing a Gaussian Process surrogate model that utilizes a composite kernel:
1.  **Structure-Aware Set Kernel**: Captures chemical similarities between sets of ligands using an optimal assignment formulation over molecular fingerprints.
2.  **Ratio Kernel**: Models the continuous effect of mixing proportions.

This framework allows for efficient learning from expensive wet-lab evaluations, guiding domain scientists toward high-performing designs with significantly fewer experiments than traditional methods.

## Installation

This project relies on `conda` for environment management. To set up the environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd structure-aware-BO
    ```

2.  **Create the Conda environment:**
    The dependencies are listed in `environment.yml`. Create the environment named `sabo`:
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate sabo
    ```

## Code Organization

The codebase is organized as follows:

- **`gp_model.py`**  
  Defines the Gaussian Process model class (`GPModel`), encompassing the prior, likelihood, and the custom structure-aware kernel integration.

- **`assignment_kernel.py`**  
  Implements the `AssignmentDistanceKernel`, which computes the similarity between sets of ligands (mixture components) by solving an optimal assignment problem based on their ground molecular similarities.

- **`mol_fingerprints.py`**  
  Contains utilities for processing chemical structures. It handles the loading of SMILES strings and the computation of Morgan fingerprints using `rdkit`.

- **`inference.py`**  
  Provides the main interface for training the GP model and running inference. It handles data loading, feature embedding using the fingerprint embedder, and model fitting.

- **`structure_aware_BO.ipynb`**  
  Demonstrates the end-to-end workflow of the framework, including data loading, data visualization, model training, and performance evaluation using the structure-aware GP.

- **`data_processing.py`**  
  Includes helper functions for cleaning raw experimental data, handling missing values, and splitting datasets for training and testing.

- **`fp_embed.py`**  
  Manages the embedding of ligand features, converting high-dimensional fingerprint data into a format suitable for the GP model.

- **`config.yaml` & `config_loader.py`**  
  Configuration files and loaders to manage global settings such as file paths, fingerprint parameters, and model hyperparameters.

- **`data/`**  
  Directory containing the dataset files (e.g., `ligand_data.csv`, `smiles.txt`) used for training and evaluation.