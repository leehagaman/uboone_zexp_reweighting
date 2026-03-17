import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from axial_form_factor_parametrizations import (
    F_A_z2, F_A_dipole,
    minerva_a_values, minerva_t0, minerva_a_universes, minerva_a_cov_matrix,
    deuterium_a_values, deuterium_t0,
    deuterium_partial_a_values, deuterium_a_cov_matrix, complete_a_values_8,
)

os.makedirs("plots", exist_ok=True)

f = uproot.open("/nevis/riverside/data/leehagaman/ngem/data_files/nu_overlay_splines_50.root")
true_q2_values_all = f["singlephotonana"]["eventweight_tree"]["GTruth_gQ2"].array(library="np")
true_scatter_values_all = f["singlephotonana"]["eventweight_tree"]["GTruth_Gscatter"].array(library="np")
true_NC_values_all = f["singlephotonana"]["eventweight_tree"]["MCTruth_neutrino_CCNC"].array(library="np")
nu_pdg_values_all = f["singlephotonana"]["eventweight_tree"]["GTruth_ProbePDG"].array(library="np")
categories = []
for true_scatter in true_scatter_values_all:
    if true_scatter == 1:
        categories.append("QE")
    elif true_scatter == 4:
        categories.append("RES")
    elif true_scatter == 3:
        categories.append("DIS")
    elif true_scatter == 10:
        categories.append("MEC")
    elif true_scatter == 5:
        categories.append("COH")
    else:
        categories.append("OTHER")
true_numuCCQE_flags = (nu_pdg_values_all == 14) & (true_NC_values_all == 0) & (true_scatter_values_all == 1)
true_q2_values = true_q2_values_all[true_numuCCQE_flags]
wc_weight_cv = f["wcpselection"]["T_eval"].arrays(["weight_cv"], library="np")["weight_cv"][true_numuCCQE_flags]
wc_weight_spline = f["wcpselection"]["T_eval"].arrays(["weight_spline"], library="np")["weight_spline"][true_numuCCQE_flags]
weights = wc_weight_cv * wc_weight_spline
weights[weights < 0] = 1
weights[weights > 30] = 1
weights[np.isnan(weights)] = 1
weights[np.isinf(weights)] = 1

q2_values = np.logspace(-2, np.log10(5), 200)

# Reference: dipole with M_A = 1.014 GeV
fa_ref = F_A_dipole(q2_values, M_A=1.014)

# Central values
fa_deuterium = F_A_z2(q2_values, deuterium_a_values, deuterium_t0)
fa_minerva = F_A_z2(q2_values, minerva_a_values, minerva_t0)

fa_dipole_0p8 = F_A_dipole(q2_values, M_A=0.8)
fa_dipole_0p9 = F_A_dipole(q2_values, M_A=0.9)
fa_dipole_1p0 = F_A_dipole(q2_values, M_A=1.0)
fa_dipole_1p1 = F_A_dipole(q2_values, M_A=1.1)
fa_dipole_1p2 = F_A_dipole(q2_values, M_A=1.2)
fa_dipole_1p3 = F_A_dipole(q2_values, M_A=1.3)
fa_dipole_1p4 = F_A_dipole(q2_values, M_A=1.4)

# MiniBooNE dipole band
miniboone_MA = 1.35
miniboone_MA_error = 0.17
fa_miniboone = F_A_dipole(q2_values, M_A=miniboone_MA)
fa_miniboone_lo = F_A_dipole(q2_values, M_A=miniboone_MA - miniboone_MA_error)
fa_miniboone_hi = F_A_dipole(q2_values, M_A=miniboone_MA + miniboone_MA_error)


# MINERvA error band from pre-computed universes
minerva_curves = np.array([F_A_z2(q2_values, a, minerva_t0) for a in minerva_a_universes])
minerva_lo = np.percentile(minerva_curves, 16, axis=0)
minerva_hi = np.percentile(minerva_curves, 84, axis=0)

# Deuterium error band from covariance matrix (symmetrize to ensure positive-semidefinite)
deuterium_a_cov_sym = (deuterium_a_cov_matrix + deuterium_a_cov_matrix.T) / 2
deuterium_partial_universes = np.random.multivariate_normal(deuterium_partial_a_values, deuterium_a_cov_sym, 1000)
deuterium_a_universes = [complete_a_values_8(p, initial_guess=[1, 1, 1, 1, 1], t0=deuterium_t0) for p in deuterium_partial_universes]
deuterium_curves = np.array([F_A_z2(q2_values, a, deuterium_t0) for a in deuterium_a_universes])
deuterium_lo = np.percentile(deuterium_curves, 16, axis=0)
deuterium_hi = np.percentile(deuterium_curves, 84, axis=0)

fig, ax = plt.subplots(figsize=(8, 6))

# deuterium
ax.plot(q2_values, fa_deuterium / fa_ref, label="Deuterium z-expansion", color="blue")
ax.fill_between(q2_values, deuterium_lo / fa_ref, deuterium_hi / fa_ref, alpha=0.2, color="blue")

# MINERvA
ax.plot(q2_values, fa_minerva / fa_ref, label="MINERvA z-expansion", color="red")
ax.fill_between(q2_values, minerva_lo / fa_ref, minerva_hi / fa_ref, alpha=0.2, color="red")

# MiniBooNE
ax.plot(q2_values, fa_miniboone / fa_ref, label=r"MiniBooNE $M_A = 1.35 \pm 0.17$ GeV", color="purple")
ax.fill_between(q2_values, fa_miniboone_lo / fa_ref, fa_miniboone_hi / fa_ref, alpha=0.2, color="purple")

# MicroBooNE Prior
ax.plot(q2_values, fa_dipole_1p1 / fa_ref, label=r"Dipole $M_A = 1.1 \pm 0.1$ GeV (uB prior)", color="k")
ax.fill_between(q2_values, fa_dipole_1p2 / fa_ref, fa_dipole_1p0 / fa_ref, alpha=0.2, color="k")

# Spline Spread
ax.plot(q2_values, fa_dipole_0p8 / fa_ref, label=r"$M_A = 0.8$ GeV", color="green")
ax.plot(q2_values, fa_dipole_1p4 / fa_ref, label=r"$M_A = 1.4$ GeV", color="green")

# MicroBooNE Q^2 distribution
ax.hist(true_q2_values, bins=np.logspace(-2, np.log10(10), 26), weights=weights*0.001, label=r"MicroBooNE $Q^2$ distribution", color="gray", alpha=0.5, zorder=-10)

ax.axhline(1, color="black", linestyle=":", alpha=0.5)

ax.set_xlabel(r"$Q^2$ (GeV$^2$)")
ax.set_xscale("log")
ax.set_ylabel(r"$F_A(Q^2)$ / Dipole $M_A = 1.014$ GeV")
ax.set_ylim(0, 4)
ax.set_title(r"Axial Form Factor Ratio to Dipole $M_A = 1.014$ GeV")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/F_A_comparison.png", dpi=150)
plt.close()

# ===========================================================================
# MicroBooNE extracted z-expansion parameters
# ===========================================================================

# PCA decomposition (same basis used when writing the spline weights)
eigenvalues, eigenvectors = np.linalg.eigh(minerva_a_cov_matrix)
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]  # columns are eigenvectors

# Best-fit PCA values and postfit covariance from PROfile ROOT file
_pfile = uproot.open("/nevis/houston/home/leehagaman/PROfit/uboone_spline_testing/Q2zexp_test_v3_v1_PROfile.root")

_gfr = _pfile["global_fit_result;1"]
_labels = _gfr.axis().labels()
_vals   = _gfr.values()
uboone_pca_values = np.array([_vals[_labels.index(f"FAzexpPCA{k}")] for k in range(1, 5)])

uboone_pca_cov  = _pfile["postfit_cov_nuisance_only;1"].values()
uboone_pca_corr = _pfile["postfit_corr_nuisance_only;1"].values()

_g = _pfile["one_sigma_errs;1"]
profile_bf     = _g.member("fY")
profile_eylow  = _g.member("fEYlow")
profile_eyhigh = _g.member("fEYhigh")

# Central-value a_partial (a1..a4 in MINERvA parameterisation)
a_partial_minerva_cv = np.array(minerva_a_values[1:5])

# Reconstruct best-fit a_partial for MicroBooNE
a_partial_uboone = a_partial_minerva_cv.copy()
for i in range(4):
    a_partial_uboone += uboone_pca_values[i] * np.sqrt(eigenvalues[i]) * eigenvectors[:, i]

uboone_a_values = complete_a_values_8(
    a_partial_uboone,
    initial_guess=minerva_a_values[0:1] + minerva_a_values[5:],
    t0=minerva_t0,
)
fa_uboone = F_A_z2(q2_values, uboone_a_values, minerva_t0)

# Uncertainty band: sample from postfit multivariate Gaussian (includes correlations)
rng = np.random.default_rng(42)
n_uni = 1000
pca_samples = rng.multivariate_normal(mean=uboone_pca_values, cov=uboone_pca_cov, size=n_uni)

uboone_curves = []
for pca_vals in pca_samples:
    a_p = a_partial_minerva_cv.copy()
    for i in range(4):
        a_p += pca_vals[i] * np.sqrt(eigenvalues[i]) * eigenvectors[:, i]
    a_full = complete_a_values_8(
        a_p,
        initial_guess=minerva_a_values[0:1] + minerva_a_values[5:],
        t0=minerva_t0,
    )
    uboone_curves.append(F_A_z2(q2_values, a_full, minerva_t0))

uboone_curves = np.array(uboone_curves)
uboone_lo = np.percentile(uboone_curves, 16, axis=0)
uboone_hi = np.percentile(uboone_curves, 84, axis=0)

# ===========================================================================
# MicroBooNE extracted M_A (dipole fit)
# ===========================================================================

# MACCQE spline: sigma=0 -> MA=1.1 GeV, sigma=1 -> MA=1.2 GeV (step = 0.1 GeV)
_MA_prior_center = 1.1   # GeV
_MA_prior_sigma  = 0.1   # GeV / sigma-unit

_mafile = uproot.open("/nevis/houston/home/leehagaman/PROfit/uboone_spline_testing/Q2MA_test_v2_v1_PROfile.root")

_ma_gfr    = _mafile["global_fit_result;1"]
_ma_labels = _ma_gfr.axis().labels()
_ma_vals   = _ma_gfr.values()
ma_uboone_sigma_bf = _ma_vals[_ma_labels.index("MACCQE")]   # in sigma units

_ma_g             = _mafile["one_sigma_errs;1"]
ma_profile_bf     = float(_ma_g.member("fY")[0])
ma_profile_eylow  = float(_ma_g.member("fEYlow")[0])
ma_profile_eyhigh = float(_ma_g.member("fEYhigh")[0])

ma_postfit_sigma  = float(np.sqrt(_mafile["postfit_cov_nuisance_only;1"].values()[0, 0]))

# Convert to GeV
ma_uboone_gev         = _MA_prior_center + ma_uboone_sigma_bf   * _MA_prior_sigma
ma_uboone_postfit_gev = ma_postfit_sigma * _MA_prior_sigma

fa_ma_uboone    = F_A_dipole(q2_values, M_A=ma_uboone_gev)
fa_ma_uboone_lo = F_A_dipole(q2_values, M_A=ma_uboone_gev - ma_uboone_postfit_gev)
fa_ma_uboone_hi = F_A_dipole(q2_values, M_A=ma_uboone_gev + ma_uboone_postfit_gev)

# ===========================================================================
# Plot: MicroBooNE extraction alongside other measurements
# ===========================================================================

fig2, ax2 = plt.subplots(figsize=(8, 6))

"""
# deuterium
ax2.plot(q2_values, fa_deuterium / fa_ref, label="Deuterium z-expansion", color="blue")
ax2.fill_between(q2_values, deuterium_lo / fa_ref, deuterium_hi / fa_ref, alpha=0.2, color="blue")
"""

# MINERvA
ax2.plot(q2_values, fa_minerva / fa_ref, label="MINERvA z-expansion (new prior)", color="red")
ax2.fill_between(q2_values, minerva_lo / fa_ref, minerva_hi / fa_ref, alpha=0.2, color="red")

"""
# MiniBooNE
ax2.plot(q2_values, fa_miniboone / fa_ref, label=r"MiniBooNE $M_A = 1.35 \pm 0.17$ GeV", color="purple")
ax2.fill_between(q2_values, fa_miniboone_lo / fa_ref, fa_miniboone_hi / fa_ref, alpha=0.2, color="purple")
"""

# MicroBooNE prior
ax2.plot(q2_values, fa_dipole_1p1 / fa_ref, label=r"Dipole $M_A = 1.1 \pm 0.1$ GeV (old prior)", color="k")
ax2.fill_between(q2_values, fa_dipole_1p2 / fa_ref, fa_dipole_1p0 / fa_ref, alpha=0.2, color="k")

# MicroBooNE extracted z-expansion
ax2.plot(q2_values, fa_uboone / fa_ref, label=r"MicroBooNE z-expansion (best fit $\pm 1\sigma$)", color="green")
ax2.fill_between(q2_values, uboone_lo / fa_ref, uboone_hi / fa_ref, alpha=0.2, color="green")

# MicroBooNE extracted M_A (dipole)
ax2.plot(q2_values, fa_ma_uboone / fa_ref,
         label=rf"MicroBooNE dipole $M_A = {ma_uboone_gev:.3f} \pm {ma_uboone_postfit_gev:.3f}$ GeV",
         color="orange")
ax2.fill_between(q2_values, fa_ma_uboone_lo / fa_ref, fa_ma_uboone_hi / fa_ref,
                 alpha=0.2, color="orange")

# MicroBooNE Q^2 distribution
ax2.hist(true_q2_values, bins=np.logspace(-2, np.log10(10), 26), weights=weights*0.001, label=r"MicroBooNE $Q^2$ distribution", color="gray", alpha=0.5, zorder=-10)

ax2.axhline(1, color="black", linestyle=":", alpha=0.5)

ax2.set_xlabel(r"$Q^2$ (GeV$^2$)")
ax2.set_xlim(1e-2, 10)
ax2.set_xscale("log")
ax2.set_ylabel(r"$F_A(Q^2)$ / Dipole $M_A = 1.014$ GeV")
ax2.set_ylim(0, 4)
ax2.set_title(r"Axial Form Factor — MicroBooNE Extraction vs Other Measurements")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/F_A_uboone_extraction.png", dpi=150)
plt.close()
print("Saved plots/F_A_uboone_extraction.png")

# ===========================================================================
# Plot: extracted curves only (no prior)
# ===========================================================================

fig2b, ax2b = plt.subplots(figsize=(8, 6))

ax2b.plot(q2_values, fa_uboone / fa_ref, label=r"MicroBooNE z-expansion (best fit $\pm 1\sigma$)", color="green")
ax2b.fill_between(q2_values, uboone_lo / fa_ref, uboone_hi / fa_ref, alpha=0.2, color="green")

ax2b.plot(q2_values, fa_ma_uboone / fa_ref,
          label=rf"MicroBooNE dipole $M_A = {ma_uboone_gev:.3f} \pm {ma_uboone_postfit_gev:.3f}$ GeV",
          color="orange")
ax2b.fill_between(q2_values, fa_ma_uboone_lo / fa_ref, fa_ma_uboone_hi / fa_ref,
                  alpha=0.2, color="orange")

# MicroBooNE Q^2 distribution
ax2b.hist(true_q2_values, bins=np.logspace(-2, np.log10(10), 26), weights=weights*0.001, label=r"MicroBooNE $Q^2$ distribution", color="gray", alpha=0.5, zorder=-10)

ax2b.axhline(1, color="black", linestyle=":", alpha=0.5)

ax2b.set_xlabel(r"$Q^2$ (GeV$^2$)")
ax2b.set_xlim(1e-2, 10)
ax2b.set_xscale("log")
ax2b.set_ylabel(r"$F_A(Q^2)$ / Dipole $M_A = 1.014$ GeV")
ax2b.set_ylim(0, 4)
ax2b.set_title(r"Axial Form Factor — MicroBooNE Extraction")
ax2b.legend()
ax2b.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/F_A_uboone_extraction_only.png", dpi=150)
plt.close()
print("Saved plots/F_A_uboone_extraction_only.png")

# ===========================================================================
# Plot: PCA parameter summary — pre-fit vs post-fit
# ===========================================================================

param_labels = [f"FAzexpPCA{k}" for k in range(1, 5)]
x = np.arange(4)
postfit_sigma = np.sqrt(np.diag(uboone_pca_cov))

fig3, ax3 = plt.subplots(figsize=(7, 5))

# Pre-fit band: prior ±1σ centred at 0
ax3.fill_between([-0.5, 3.5], -1, 1, color="gray", alpha=0.2, label="Pre-fit ±1σ (prior)")
ax3.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

# Post-fit band: ±postfit σ centred on global best-fit
ax3.bar(x, 2 * postfit_sigma, bottom=uboone_pca_values - postfit_sigma,
        width=0.5, color="steelblue", alpha=0.4, label="Post-fit ±1σ", zorder=2)

# Profile scan best-fit with asymmetric 1σ error bars
ax3.errorbar(x - 0.12, profile_bf,
             yerr=[profile_eylow, profile_eyhigh],
             fmt='o', color="black", capsize=4, markersize=5,
             label="Profile scan best-fit ±1σ", zorder=4)

# Global best-fit points
ax3.plot(x + 0.12, uboone_pca_values, 's', color="red", markersize=7,
         label="Global best-fit", zorder=5)

ax3.set_xticks(x)
ax3.set_xticklabels(param_labels)
ax3.set_xlim(-0.5, 3.5)
ax3.set_ylabel("Parameter value (prior σ = 1)")
ax3.set_title("FAzexp PCA Parameter Fit Results")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("plots/FA_pca_params_summary.png", dpi=150)
plt.close()
print("Saved plots/FA_pca_params_summary.png")

# ===========================================================================
# Plot: Post-fit correlation matrix
# ===========================================================================

fig4, ax4 = plt.subplots(figsize=(5, 4))

im = ax4.imshow(uboone_pca_corr, vmin=-1, vmax=1, cmap="RdBu_r")
fig4.colorbar(im, ax=ax4, label="Correlation")

ax4.set_xticks(range(4))
ax4.set_yticks(range(4))
ax4.set_xticklabels(param_labels, rotation=45, ha="right")
ax4.set_yticklabels(param_labels)

for i in range(4):
    for j in range(4):
        ax4.text(j, i, f"{uboone_pca_corr[i, j]:.3f}",
                 ha="center", va="center", fontsize=9,
                 color="white" if abs(uboone_pca_corr[i, j]) > 0.5 else "black")

ax4.set_title("FAzexp PCA Post-fit Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/FA_pca_postfit_corr.png", dpi=150)
plt.close()
print("Saved plots/FA_pca_postfit_corr.png")

# ===========================================================================
# Plot: M_A parameter summary — pre-fit vs post-fit
# ===========================================================================

fig5, ax5 = plt.subplots(figsize=(4, 5))

# Pre-fit band: prior ±1σ centred at 0
ax5.fill_between([-0.5, 0.5], -1, 1, color="gray", alpha=0.2, label="Pre-fit ±1σ (prior)")
ax5.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

# Post-fit band: ±postfit σ centred on global best-fit
ax5.bar([0], 2 * ma_postfit_sigma, bottom=ma_uboone_sigma_bf - ma_postfit_sigma,
        width=0.5, color="orange", alpha=0.4, label="Post-fit ±1σ", zorder=2)

# Profile scan best-fit with asymmetric 1σ error bars
ax5.errorbar([-0.12], [ma_profile_bf],
             yerr=[[ma_profile_eylow], [ma_profile_eyhigh]],
             fmt='o', color="black", capsize=4, markersize=5,
             label="Profile scan best-fit ±1σ", zorder=4)

# Global best-fit point
ax5.plot([0.12], [ma_uboone_sigma_bf], 's', color="red", markersize=7,
         label="Global best-fit", zorder=5)

ax5.set_xticks([0])
ax5.set_xticklabels(["MACCQE"])
ax5.set_xlim(-0.5, 0.5)
ax5.set_ylabel("Parameter value (prior σ = 1)")
ax5.set_title("$M_A$ Dipole Parameter Fit Results")
ax5.legend()
ax5.grid(True, alpha=0.3, axis="y")

# Secondary y-axis in GeV
ax5_gev = ax5.secondary_yaxis(
    "right",
    functions=(lambda s: _MA_prior_center + s * _MA_prior_sigma,
               lambda m: (m - _MA_prior_center) / _MA_prior_sigma),
)
ax5_gev.set_ylabel(r"$M_A$ (GeV)")

plt.tight_layout()
plt.savefig("plots/MA_param_summary.png", dpi=150)
plt.close()
print("Saved plots/MA_param_summary.png")
