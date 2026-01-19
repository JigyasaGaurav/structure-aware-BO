import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch.utils.transforms import normalize
from gp_model import GPModel
from config_loader import SMILES_FILE, MODEL_PATH, PLOT_DATA_PATH, PLOT_PATH, TARGET_COLUMNS


def cv_plot(q_middle, q_lower, q_upper, Y_true, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    min_val = (torch.minimum(Y_true, q_lower)).min()
    max_val = (torch.maximum(Y_true, q_upper)).max()
    min_val, max_val = min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val)
    ax.plot([min_val, max_val], [min_val, max_val], "b--", lw=2)

    yerr1, yerr2 = q_middle - q_lower, q_upper - q_middle
    yerr = torch.cat((yerr1.unsqueeze(0), yerr2.unsqueeze(0)), dim=0).squeeze(-1)
    markers, caps, bars = ax.errorbar(
        Y_true.cpu().numpy(),
        q_middle.cpu().numpy(),
        yerr=yerr.cpu().numpy(),
        fmt=".",
        capsize=4,
        elinewidth=2.0,
        ms=14,
        c="k",
        ecolor="gray",
    )
    [bar.set_alpha(0.8) for bar in bars]
    [cap.set_alpha(0.8) for cap in caps]
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.grid(True)

def cv_plots(gp_model, test_x, test_y):    
    with torch.no_grad():
        posterior = gp_model.posterior(test_x)
        mean = posterior.mean
        var = posterior.variance
        
        mean = mean.detach().squeeze()
        print(f"mean {mean}")
        var = var.detach().squeeze()
        q1 = mean - 1.96 * var.sqrt()
        q2 = mean + 1.96 * var.sqrt()
        
        cv_plot(q_middle=mean, q_lower=q1, q_upper=q2, Y_true=test_y)
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")

if __name__ == "__main__":
    df = pd.read_csv(PLOT_DATA_PATH)
    X = df[['Ligand 1', 'Ligand 2', 'n_1', 'n_2']].copy()
    for col in ['n_1', 'n_2']:
        X[col] = X[col] / 24.0

    gp = GPModel(target_column=TARGET_COLUMNS[0], smiles_path=SMILES_FILE)
    gp.load_model(MODEL_PATH, X_template=X)

    X_encoded = gp._encode_features(X)
    X_tensor = torch.tensor(X_encoded[gp.feature_columns].values, dtype=torch.double)
    y = torch.tensor(df[TARGET_COLUMNS[0]].values, dtype=torch.double)

    cv_plots(gp, X_tensor, y)
    plt.title('GP Model Predictions with 95% Confidence Intervals')
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
