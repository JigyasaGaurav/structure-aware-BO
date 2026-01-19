import csv
from itertools import combinations, product

def generate_combinations():
    ligands = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
    
    # All unique ligand pairs (no permutation, no self-combinations)
    ligand_pairs = list(combinations(ligands, 2))
    all_combinations = []
    
    for ligand1, ligand2 in ligand_pairs:
        for ratio1 in range(1, 24):
            ratio2 = 24 - ratio1
            all_combinations.append({
                'Ligand 1': ligand1,
                'n_1': ratio1,
                'Ligand 2': ligand2,
                'n_2': ratio2
            })
    
    return all_combinations

def save_to_csv(combinations, filename='data/all_candidates.csv'):
    fieldnames = ['Ligand 1', 'n_1', 'Ligand 2', 'n_2']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combinations)
    
    print(f"Saved {len(combinations)} combinations to {filename}")

if __name__ == "__main__":
    all_combs = generate_combinations()
    print(len(all_combs), "combinations generated.")
    save_to_csv(all_combs)
    
    ligands = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
    num_ligands = len(ligands)
    num_pairs = num_ligands * (num_ligands - 1) // 2  # C(17,2)
    num_ratio_combinations = 23
    total_combinations = num_pairs * num_ratio_combinations
    
    print(f"Number of ligands: {num_ligands}")
    print(f"Number of unique pairs: {num_pairs}")
    print(f"Number of ratio combinations per pair: {num_ratio_combinations}")
    print(f"Total combinations: {total_combinations}")