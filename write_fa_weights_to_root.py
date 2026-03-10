"""
Copy nu_overlay_splines_50.root → nu_overlay_splines_50_FAweights.root and
add weight_minerva_FA and weight_spline_FAzexpPCA{1-4} branches to the
spline_weights tree.

All 37,760 events are processed (no numuCCQE filter), so the new branches
cover the full tree identically to the other weight branches already there.
"""

import os
import shutil
import array
import uproot
import numpy as np
import ROOT
from axial_form_factor_parametrizations import (
    F_A_z2, get_MA_effective, get_weight,
    minerva_a_values, minerva_t0, minerva_a_cov_matrix, complete_a_values_8,
)

SRC = "/nevis/riverside/data/leehagaman/ngem/data_files/nu_overlay_splines_50.root"
DST = "/nevis/riverside/data/leehagaman/ngem/data_files/nu_overlay_splines_50_FAweights.root"

# ============================================================
# 1.  Copy the ROOT file first
# ============================================================
if os.path.exists(DST):
    print(f"Removing existing {DST}...")
    os.remove(DST)

print(f"Copying {SRC} → {DST}  (4+ GB, may take a few minutes)...")
shutil.copy2(SRC, DST)
print("Copy complete.")

# ============================================================
# 2.  Read everything needed from the copy with uproot
# ============================================================
print("Reading data from copied file...")
f = uproot.open(DST)

weight_spline_all = f["wcpselection/T_eval"]["weight_spline"].array(library="np")
true_q2_all       = f["singlephotonana/eventweight_tree"]["GTruth_gQ2"].array(library="np")

MA_spline_raw = f["spline_weights"]["MaCCQE_UBGenie"].array(library="np")
N = len(true_q2_all)
MA_grid = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])

print(f"  {N} events loaded.")

all_MA_weights = np.column_stack([
    np.array([x[j] for x in MA_spline_raw])
    for j in range(7)
])  # shape (N, 7)
for j in range(7):
    col = all_MA_weights[:, j]
    col[np.isnan(col)] = 1
    col[np.isinf(col)] = 1

del MA_spline_raw  # free memory

# ============================================================
# 2.  Compute MINERvA z-expansion CV weight (all events)
# ============================================================
print("Computing MINERvA CV weights...")
FA_cv       = F_A_z2(true_q2_all, minerva_a_values, minerva_t0)
MA_eff_cv   = get_MA_effective(true_q2_all, FA_cv)
w_zexp_cv   = get_weight(MA_eff_cv, MA_grid, all_MA_weights)
weight_minerva_FA_all = weight_spline_all * w_zexp_cv  # shape (N,)

# ============================================================
# 3.  PCA decomposition and sigma variations (all events)
# ============================================================
SIGMA_VALUES = np.array([-3, -2, -1, 0, 1, 2, 3])
a_partial_cv = np.array(minerva_a_values[1:5])

eigenvalues, eigenvectors = np.linalg.eigh(minerva_a_cov_matrix)
sort_idx     = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]

print("Computing PCA sigma variations (all events)...")
pca_weights_all = []   # list of 4 arrays, each shape (N, 7)
for i in range(4):
    shift = np.sqrt(eigenvalues[i]) * eigenvectors[:, i]
    cols = []
    for s in SIGMA_VALUES:
        if s == 0:
            cols.append(weight_minerva_FA_all)
        else:
            a_partial = a_partial_cv + s * shift
            a_full = complete_a_values_8(
                a_partial,
                initial_guess=minerva_a_values[0:1] + minerva_a_values[5:],
                t0=minerva_t0,
            )
            FA   = F_A_z2(true_q2_all, a_full, minerva_t0)
            MA_e = get_MA_effective(true_q2_all, FA)
            w    = get_weight(MA_e, MA_grid, all_MA_weights)
            cols.append(weight_spline_all * w)
    pca_weights_all.append(np.column_stack(cols))  # (N, 7)
print("Done computing weights.")

# ============================================================
# 4.  Open copy in UPDATE mode and add new branches
# ============================================================
print("Opening copy with ROOT and adding branches...")
tfile = ROOT.TFile(DST, "UPDATE")
if tfile.IsZombie():
    raise RuntimeError(f"Could not open {DST} for update")

tree = tfile.Get("spline_weights")
if not tree:
    raise RuntimeError("Could not find spline_weights tree")

N_SIGMA = len(SIGMA_VALUES)  # 7

# Branch buffers
buf_minerva = array.array('f', [0.0])
buf_pca     = [array.array('f', [0.0] * N_SIGMA) for _ in range(4)]

b_minerva = tree.Branch(
    "weight_minerva_FA", buf_minerva, "weight_minerva_FA/F"
)
b_pca = [
    tree.Branch(
        f"weight_spline_FAzexpPCA{i+1}",
        buf_pca[i],
        f"weight_spline_FAzexpPCA{i+1}[{N_SIGMA}]/F",
    )
    for i in range(4)
]

print(f"Filling {N} entries...")
for ev in range(N):
    buf_minerva[0] = float(weight_minerva_FA_all[ev])
    b_minerva.Fill()

    for i in range(4):
        for k in range(N_SIGMA):
            buf_pca[i][k] = float(pca_weights_all[i][ev, k])
        b_pca[i].Fill()

    if (ev + 1) % 5000 == 0:
        print(f"  {ev+1}/{N}")

print("Writing updated tree...")
tree.Write("", ROOT.TObject.kOverwrite)
tfile.Close()
print(f"Done. New file: {DST}")
