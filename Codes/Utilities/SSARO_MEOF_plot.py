
# SSARO MEOF vs robust reference ND set plot
import os
import numpy as np
import matplotlib.pyplot as plt

# SSARO result folders
problem_result_folder = [
    "RTP1_mean",
    "RTP2_mean",
    "TP4_mean",
]

# Problem names used by the NSGA-II files
problem = [
    "RTP1_reference_100",
    "RTP2_reference_100",
    "TP4_reference_100",
]

# Median run for each problem
median_runs = [
    25,   # RTP1
    21,   # RTP2
    8,   # TP4
]

save_folder = "Median_HV_figure_MEOF"
os.makedirs(save_folder, exist_ok=True)

for result_folder, prob, median_run in zip(problem_result_folder,
                                           problem,
                                           median_runs):

    print(f"Processing {prob}...")

    # Load data
    Rnd_SSARO = np.load(
        fr"Results_MEOF\{result_folder}\Robust_ND_front_points\robust_nd_front_true_run_{median_run:02d}.npy"
    )

    Rnd = np.load(f"MEOF robust reference_set\{prob}.npy")
    # Pnd = np.load(f"Performance_front_NSGA-II/Performance_front_{prob}_F.npy")
    Pred_Rnd = np.load(
        fr"Results_MEOF\{result_folder}\Robust_ND_front_points\robust_nd_front_predicted_run_{median_run:02d}.npy"
        )

    # Plot
    plt.figure(figsize=(7, 5))

    plt.scatter(
        Rnd[:, 0],
        Rnd[:, 1],
        label="MEOF robust reference set",
        c="red",
        marker="o",
        s=45,
    )

    plt.scatter(
        Rnd_SSARO[:, 0],
        Rnd_SSARO[:, 1],
        label="SSARO - MEOF robust ND front",
        c="black",
        marker="*",
        s=90,
    )

    plt.xlabel(r"$f_1$", fontsize=20)
    plt.ylabel(r"$f_2$", fontsize=20)
    # plt.title(prob)
    plt.tick_params(axis="both", labelsize=16, length=6, width=1.2)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(
        save_folder,
        f"{result_folder}_median_HV.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

print("\nFinished.")