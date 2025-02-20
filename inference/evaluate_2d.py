import matplotlib.pylab as plt
import numpy as np
import tqdm
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from dataset.apps.evaluate_solver import *

def evaluate(root, folder, n_sim=50):
    path = os.path.join(root, "inference_results", folder)
    smoke_out_dir = os.path.join(path, "smoke_outs")
    smoke_out_files = os.listdir(smoke_out_dir)
    n_sim = len(smoke_out_files)
    if n_sim == 0:
        return
    smoke_out_files.sort()
    all_smoke_out = []
    for i in range(n_sim):
        smoke = np.load(os.path.join(smoke_out_dir, "{}.npy".format(i)))
        final_smoke_out = smoke[-1]
        all_smoke_out.append(final_smoke_out)
    avg_smoke_out = np.mean(all_smoke_out, axis=0)
    print(n_sim, folder, ", avg_smoke_out: ", avg_smoke_out, ", J: ", 1-avg_smoke_out, "\n")

if __name__ == '__main__':
    root = "/data/close_loop_diffcon/phiflow/"
    n_sim = 50
    all_folders = os.listdir(os.path.join(root, "inference_results"))
    all_folders.sort()
    for folder in all_folders:
        evaluate(root, folder, n_sim)