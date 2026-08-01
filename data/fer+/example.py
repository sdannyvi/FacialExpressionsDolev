import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/gpfs0/bgu-vilenchi/users/sdolev')
from Thesis.VLMs.data_cleaning.remove_duplicates import vis_res_eps


eps_list = [0.11, 0.09, 0.07, 0.05, 0.03, 0.01]
npz_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_clustering_embeddings.npz"

npz_file = np.load(npz_path, allow_pickle=True)

plots_list = vis_res_eps(res_npz=npz_file, eps_list=eps_list, n_clusters_to_sample=5, seed=4)

out_dir = "figures"
os.makedirs(out_dir, exist_ok=True)

for i, fig in enumerate(plots_list):
    save_path = os.path.join(out_dir, f"clusters_eps_{i}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
