"""
Add weight_minerva_FA and weight_spline_FAzexpPCA{1-4} branches to the
spline_weights tree of a ROOT file.

Usage
-----
    python write_fa_weights_to_root.py input.root [output.root] [options]

If output.root is omitted the script writes to
<input_stem>_FAweights.root in the same directory.
"""

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Add FA z-expansion weights to a ROOT file."
    )
    p.add_argument("src", help="Input ROOT file")
    p.add_argument(
        "dst", nargs="?", default=None,
        help="Output ROOT file (default: <src_stem>_FAweights.root)",
    )
    p.add_argument(
        "--weight-spline-tree",
        default="wcpselection/T_eval",
        help="Tree path for weight_spline branch (default: wcpselection/T_eval)",
    )
    p.add_argument(
        "--q2-tree",
        default="singlephotonana/eventweight_tree",
        help="Tree path for GTruth_gQ2 branch (default: singlephotonana/eventweight_tree)",
    )
    p.add_argument(
        "--ma-spline-tree",
        default="spline_weights",
        help="Tree path for MaCCQE_UBGenie branch (default: spline_weights)",
    )
    p.add_argument(
        "--output-tree",
        default="spline_weights",
        help="Tree to add the new branches to (default: spline_weights)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    SRC = args.src
    if args.dst is None:
        stem, ext = os.path.splitext(SRC)
        DST = stem + "_FAweights" + ext
    else:
        DST = args.dst

    # ============================================================
    # 1.  Copy the ROOT file first
    # ============================================================
    if os.path.exists(DST):
        print(f"Removing existing {DST}...")
        os.remove(DST)

    print(f"Copying {SRC} → {DST}  (may take a few minutes)...")
    shutil.copy2(SRC, DST)
    print("Copy complete.")

    # ============================================================
    # 2.  Read everything needed from the copy with uproot
    # ============================================================
    print("Reading data from copied file...")
    f = uproot.open(DST)

    weight_spline_all = f[args.weight_spline_tree]["weight_spline"].array(library="np")
    true_q2_all       = f[args.q2_tree]["GTruth_gQ2"].array(library="np")

    MA_spline_raw = f[args.ma_spline_tree]["MaCCQE_UBGenie"].array(library="np")
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
    # 3.  Compute MINERvA z-expansion CV weight (all events)
    # ============================================================
    print("Computing MINERvA CV weights...")
    FA_cv       = F_A_z2(true_q2_all, minerva_a_values, minerva_t0)
    MA_eff_cv   = get_MA_effective(true_q2_all, FA_cv)
    w_zexp_cv   = get_weight(MA_eff_cv, MA_grid, all_MA_weights)
    weight_minerva_FA_all = w_zexp_cv  # shape (N,) — analogous to MaCCQE_UBGenie[3], does NOT include weight_spline

    # ============================================================
    # 4.  PCA decomposition and sigma variations (all events)
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
                cols.append(w)
        pca_weights_all.append(np.column_stack(cols))  # (N, 7)
    print("Done computing weights.")

    # ============================================================
    # 5.  Open copy in UPDATE mode and add new branches
    # ============================================================
    print("Opening copy with ROOT and adding branches...")
    tfile = ROOT.TFile(DST, "UPDATE")
    if tfile.IsZombie():
        raise RuntimeError(f"Could not open {DST} for update")

    tree = tfile.Get(args.output_tree)
    if not tree:
        raise RuntimeError(f"Could not find tree '{args.output_tree}'")

    N_SIGMA = len(SIGMA_VALUES)  # 7

    # Branch buffers
    buf_minerva = array.array('f', [0.0])
    buf_pca     = [ROOT.std.vector('float')() for _ in range(4)]

    b_minerva = tree.Branch(
        "weight_minerva_FA", buf_minerva, "weight_minerva_FA/F"
    )
    b_pca = [
        tree.Branch(f"weight_spline_FAzexpPCA{i+1}", buf_pca[i])
        for i in range(4)
    ]

    print(f"Filling {N} entries...")
    for ev in range(N):
        buf_minerva[0] = float(weight_minerva_FA_all[ev])
        b_minerva.Fill()

        for i in range(4):
            buf_pca[i].clear()
            for k in range(N_SIGMA):
                buf_pca[i].push_back(float(pca_weights_all[i][ev, k]))
            b_pca[i].Fill()

        if (ev + 1) % 5000 == 0:
            print(f"  {ev+1}/{N}")

    print("Writing updated tree...")
    tree.Write("", ROOT.TObject.kOverwrite)
    tfile.Close()
    print(f"Done. New file: {DST}")


if __name__ == "__main__":
    main()
