"""
Reproduce the plots from build_event_dataframe.py using only data from
nu_overlay_splines_50_FAweights.root.  No recomputation of z-expansion
weights is needed for Plot 1 or the PCA universes; the covariance-universe
plot still samples from the MINERvA covariance matrix but uses the
MaCCQE_UBGenie spline weights already in the file.
"""

import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from axial_form_factor_parametrizations import (
    F_A_z2, get_MA_effective, get_weight,
    minerva_a_values, minerva_t0, minerva_a_cov_matrix, complete_a_values_8,
)

os.makedirs("plots", exist_ok=True)

ROOT_FILE = "/nevis/riverside/data/leehagaman/ngem/data_files/nu_overlay_splines_50_FAweights.root"

# ============================================================
# Load data (single file, all trees aligned by index)
# ============================================================
print("Reading data...")
f = uproot.open(ROOT_FILE)

true_q2_all       = f["singlephotonana/eventweight_tree"]["GTruth_gQ2"].array(library="np")
true_scatter_all  = f["singlephotonana/eventweight_tree"]["GTruth_Gscatter"].array(library="np")
true_NC_all       = f["singlephotonana/eventweight_tree"]["MCTruth_neutrino_CCNC"].array(library="np")
nu_pdg_all        = f["singlephotonana/eventweight_tree"]["GTruth_ProbePDG"].array(library="np")

true_numuCCQE = (nu_pdg_all == 14) & (true_NC_all == 0) & (true_scatter_all == 1)

def sel(arr):
    return arr[true_numuCCQE]

true_q2       = sel(true_q2_all)
weight_spline = sel(f["wcpselection/T_eval"]["weight_spline"].array(library="np"))

# MaCCQE_UBGenie: jagged array of 7 values per event
MA_spline_raw = f["spline_weights"]["MaCCQE_UBGenie"].array(library="np")
MA_grid       = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
all_MA_weights = np.column_stack([
    sel(np.array([x[j] for x in MA_spline_raw]))
    for j in range(7)
])  # shape (N, 7)
for j in range(7):
    col = all_MA_weights[:, j]
    col[np.isnan(col)] = 1
    col[np.isinf(col)] = 1

# New FA weight branches (scalar and fixed-size array)
weight_minerva_FA = sel(f["spline_weights"]["weight_minerva_FA"].array(library="np"))

SIGMA_VALUES = np.array([-3, -2, -1, 0, 1, 2, 3])
pca_weights = []
for i in range(1, 5):
    raw = f["spline_weights"][f"weight_spline_FAzexpPCA{i}"].array(library="np")
    # raw is shape (N_all, 7) — fixed-size array branch
    pca_weights.append(sel(raw))  # (N_sel, 7)

print(f"  {len(true_q2)} numuCCQE events selected.")

# ============================================================
# Shared helpers
# ============================================================
Q2_bins = np.logspace(-2, 2, 41)

def ratio_hist(weights_num, weights_denom, q2, bins):
    n_num, _ = np.histogram(q2, bins=bins, weights=weights_num)
    n_den, _ = np.histogram(q2, bins=bins, weights=weights_denom)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_den > 0, n_num / n_den, np.nan)

def hist_band(x, weights_list, bins):
    uni_hists = np.array([np.histogram(x, bins=bins, weights=w)[0] for w in weights_list])
    return np.percentile(uni_hists, 16, axis=0), np.percentile(uni_hists, 84, axis=0)

def frac_half_width(n_cv, n_lo, n_hi):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_cv > 0, (n_hi - n_lo) / (2 * n_cv), 0.0)

# ============================================================
# Plot 1: variations vs true Q^2
# ============================================================
MA_target         = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
MA_target_indices = [0,   1,   2,   3,   4,   5  ]
w_ref_MA = weight_spline * all_MA_weights[:, 3]   # MA=1.1

fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True,
                         gridspec_kw={"hspace": 0.08})

# --- Top panel: MA variations ---
ax_ma = axes[0]
MA_COLORS = plt.cm.RdYlBu_r(np.linspace(0.05, 0.95, 6))
for ma_val, idx, color in zip(MA_target, MA_target_indices, MA_COLORS):
    r = ratio_hist(weight_spline * all_MA_weights[:, idx], w_ref_MA, true_q2, Q2_bins)
    ax_ma.stairs(r, Q2_bins, color=color, label=rf"$M_A={ma_val:.1f}$", linewidth=1.5)
r_min = ratio_hist(weight_minerva_FA, w_ref_MA, true_q2, Q2_bins)
ax_ma.stairs(r_min, Q2_bins, color="black", linewidth=1.5, linestyle="--",
             label="MINERvA z-exp. CV")
ax_ma.axhline(1, color="gray", linestyle=":", alpha=0.6)
ax_ma.set_ylabel(r"Ratio to $M_A=1.1$")
ax_ma.set_ylim(0.5, 1.5)
ax_ma.legend(fontsize=7, ncol=4, loc="upper right")
ax_ma.set_title(r"Axial FF variations vs true $Q^2$")

# --- PCA panels ---
SIGMA_COLORS = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(SIGMA_VALUES)))
for i in range(4):
    ax = axes[i + 1]
    w_pca_i = pca_weights[i]   # (N, 7)
    for k, (s, color) in enumerate(zip(SIGMA_VALUES, SIGMA_COLORS)):
        r = ratio_hist(w_pca_i[:, k], weight_minerva_FA, true_q2, Q2_bins)
        ax.stairs(r, Q2_bins, color=color,
                  linestyle="--" if s == 0 else "-",
                  linewidth=2.0 if s == 0 else 1.2,
                  label=rf"$\sigma={s:+d}$")
    ax.axhline(1, color="gray", linestyle=":", alpha=0.6)
    ax.set_ylabel("Ratio to MINERvA CV")
    ax.set_ylim(0.5, 1.5)
    ax.legend(fontsize=7, ncol=4, loc="upper right", title=f"PCA component {i+1}")

axes[-1].set_xlabel(r"True $Q^2$ (GeV$^2$)")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(Q2_bins[0], Q2_bins[-1])

plt.savefig("plots/variations_vs_true_Q2_fromroot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/variations_vs_true_Q2_fromroot.png")

# ============================================================
# Plot 2: fractional uncertainty comparison
# ============================================================
N_UNI = 500
rng   = np.random.default_rng(42)

# --- Dipole MA=1.1 ± 0.1 ---
n_cv_MA, _ = np.histogram(true_q2, bins=Q2_bins,
                           weights=weight_spline * all_MA_weights[:, 3])
n_lo_MA, _ = np.histogram(true_q2, bins=Q2_bins,
                           weights=weight_spline * all_MA_weights[:, 2])
n_hi_MA, _ = np.histogram(true_q2, bins=Q2_bins,
                           weights=weight_spline * all_MA_weights[:, 4])
frac_MA = frac_half_width(n_cv_MA, n_lo_MA, n_hi_MA)

# --- 500 PCA universes (all data from ROOT file) ---
# For each component compute ratio = w_PCA_i / weight_minerva_FA; then per
# universe sample sigma_i ~ N(0,1), interpolate each ratio, multiply together.
print("Generating PCA universes...")
pca_ratios = []
for i in range(4):
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_i = np.where(
            weight_minerva_FA[:, None] > 0,
            pca_weights[i] / weight_minerva_FA[:, None],
            1.0,
        )
    pca_ratios.append(ratio_i)

sigmas_all = rng.standard_normal((N_UNI, 4))
pca_universe_weights = []
for u in range(N_UNI):
    combined_ratio = np.ones(len(true_q2))
    for i in range(4):
        s = sigmas_all[u, i]
        idx_lo = int(np.clip(np.floor(s + 3), 0, 5))
        idx_hi = idx_lo + 1
        t = (s - SIGMA_VALUES[idx_lo]) / (SIGMA_VALUES[idx_hi] - SIGMA_VALUES[idx_lo])
        combined_ratio *= pca_ratios[i][:, idx_lo] * (1 - t) + pca_ratios[i][:, idx_hi] * t
    pca_universe_weights.append(weight_minerva_FA * combined_ratio)
print("Done.")

n_cv_min, _ = np.histogram(true_q2, bins=Q2_bins, weights=weight_minerva_FA)
n_lo_pca, n_hi_pca = hist_band(true_q2, pca_universe_weights, Q2_bins)
frac_pca_uni = frac_half_width(n_cv_min, n_lo_pca, n_hi_pca)

# --- 500 MINERvA covariance universes ---
# Resample from the covariance matrix; use all_MA_weights from the ROOT file.
print("Generating MINERvA covariance universes...")
a_partial_cv = np.array(minerva_a_values[1:5])
cov_samples  = rng.multivariate_normal(a_partial_cv, minerva_a_cov_matrix, N_UNI)
cov_universe_weights = []
for u in range(N_UNI):
    a_full = complete_a_values_8(
        cov_samples[u],
        initial_guess=minerva_a_values[0:1] + minerva_a_values[5:],
        t0=minerva_t0,
    )
    FA  = F_A_z2(true_q2, a_full, minerva_t0)
    MAe = get_MA_effective(true_q2, FA)
    w   = get_weight(MAe, MA_grid, all_MA_weights)
    cov_universe_weights.append(weight_spline * w)
print("Done.")

n_lo_cov, n_hi_cov = hist_band(true_q2, cov_universe_weights, Q2_bins)
frac_cov_uni = frac_half_width(n_cv_min, n_lo_cov, n_hi_cov)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.stairs(frac_MA,      Q2_bins, color="steelblue",
          label=r"Dipole $M_A = 1.1 \pm 0.1$ GeV", linewidth=1.5)
ax.stairs(frac_pca_uni, Q2_bins, color="C2",
          label="PCA universes (500, independent Gaussian per component)", linewidth=1.5)
ax.stairs(frac_cov_uni, Q2_bins, color="C3", linestyle="--",
          label="MINERvA covariance universes (500)", linewidth=1.5)
ax.set_xscale("log")
ax.set_xlim(Q2_bins[0], Q2_bins[-1])
ax.set_ylim(0, 0.3)
ax.set_xlabel(r"True $Q^2$ (GeV$^2$)")
ax.set_ylabel("Fractional uncertainty (half-width, 16th/84th pctile)")
ax.set_title("Fractional uncertainty comparison")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("plots/fractional_uncertainty_comparison_fromroot.png", dpi=150)
plt.close()
print("Saved plots/fractional_uncertainty_comparison_fromroot.png")
