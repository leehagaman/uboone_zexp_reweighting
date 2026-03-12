
import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from axial_form_factor_parametrizations import (
    F_A_z2, get_MA_effective, get_weight,
    minerva_a_values, minerva_t0, minerva_a_cov_matrix, complete_a_values_8,
)

os.makedirs("plots", exist_ok=True)

f = uproot.open("/nevis/riverside/data/leehagaman/ngem/data_files/nu_overlay_splines_50.root")

true_q2_values_all = f["singlephotonana"]["eventweight_tree"]["GTruth_gQ2"].array(library="np")
true_scatter_values_all = f["singlephotonana"]["eventweight_tree"]["GTruth_Gscatter"].array(library="np")
true_NC_values_all = f["singlephotonana"]["eventweight_tree"]["MCTruth_neutrino_CCNC"].array(library="np")
nu_pdg_values_all = f["singlephotonana"]["eventweight_tree"]["GTruth_ProbePDG"].array(library="np")

true_numuCCQE_flags = (nu_pdg_values_all == 14) & (true_NC_values_all == 0) & (true_scatter_values_all == 1)

true_q2_values = true_q2_values_all[true_numuCCQE_flags]
wc_energies = f["wcpselection"]["T_KINEvars"].arrays("kine_reco_Enu", library="np")["kine_reco_Enu"][true_numuCCQE_flags]

# reco muon 4-momentum [px, py, pz, E] in GeV from T_PFeval
reco_muon_p4 = f["wcpselection"]["T_PFeval"].arrays(["reco_muonMomentum"], library="np")["reco_muonMomentum"][true_numuCCQE_flags]
mu_px, mu_py, mu_pz, mu_E = reco_muon_p4[:, 0], reco_muon_p4[:, 1], reco_muon_p4[:, 2], reco_muon_p4[:, 3]
mu_p_mag = np.sqrt(mu_px**2 + mu_py**2 + mu_pz**2)
with np.errstate(invalid="ignore", divide="ignore"):
    cos_theta_mu = np.where(mu_p_mag > 0, mu_pz / mu_p_mag, np.nan)

wc_weight_cv = f["wcpselection"]["T_eval"].arrays(["weight_cv"], library="np")["weight_cv"][true_numuCCQE_flags]
wc_weight_spline = f["wcpselection"]["T_eval"].arrays(["weight_spline"], library="np")["weight_spline"][true_numuCCQE_flags]

weights = wc_weight_cv * wc_weight_spline
weights[weights < 0] = 1
weights[weights > 30] = 1
weights[np.isnan(weights)] = 1
weights[np.isinf(weights)] = 1

MA_spline_weights = f["spline_weights"].arrays(["MaCCQE_UBGenie"], library="np")["MaCCQE_UBGenie"]

# indices 0-6 correspond to MA = 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4
MA_grid = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
all_MA_weights = np.column_stack([
    np.array([x[j] for x in MA_spline_weights])[true_numuCCQE_flags]
    for j in range(7)
])  # shape (N_events, 7)
for j in range(7):
    col = all_MA_weights[:, j]
    col[np.isnan(col)] = 1
    col[np.isinf(col)] = 1

# Event weights for all 7 MA values; indices 0-6 → MA = 0.8 ... 1.4
all_dipole_event_weights = [wc_weight_spline * all_MA_weights[:, j] for j in range(7)]

# AxFFCCQEshape weights: index 0 = dipole, index 1 = z-expansion shape
axff_raw = f["spline_weights"].arrays(["AxFFCCQEshape_UBGenie"], library="np")["AxFFCCQEshape_UBGenie"]
all_axff_weights = np.column_stack([
    np.array([x[j] for x in axff_raw])[true_numuCCQE_flags]
    for j in range(2)
])  # shape (N_events, 2)
for j in range(2):
    col = all_axff_weights[:, j]
    col[np.isnan(col)] = 1
    col[np.isinf(col)] = 1

axff_event_weights = [wc_weight_spline * all_axff_weights[:, j] for j in range(2)]

# --- M_A = 1.1 ± 0.1 dipole weights (for uncertainty band) ---
# lo: MA=1.0 (index 2), cv: MA=1.1 (index 3), hi: MA=1.2 (index 4)
dipole_weights_cv = all_dipole_event_weights[3]
dipole_weights_lo = all_dipole_event_weights[2]
dipole_weights_hi = all_dipole_event_weights[4]

# --- MINERvA z-expansion central weights ---
FA_minerva_cv = F_A_z2(true_q2_values, minerva_a_values, minerva_t0)
MA_eff_cv = get_MA_effective(true_q2_values, FA_minerva_cv)
minerva_zexp_weights_cv = get_weight(MA_eff_cv, MA_grid, all_MA_weights)
minerva_weights_cv = wc_weight_spline * minerva_zexp_weights_cv

# --- MINERvA z-expansion universe weights (from 4x4 covariance matrix) ---
# Generate 1000 universes by sampling a1-a4 from the covariance matrix,
# then completing the full 9-parameter set via the sum rules.
print("Generating MINERvA universes and computing weights...")
rng = np.random.default_rng(42)
minerva_partial_universes = rng.multivariate_normal(
    minerva_a_values[1:5], minerva_a_cov_matrix, 1000
)
minerva_universe_event_weights = []
for partial_a in minerva_partial_universes:
    a_uni = complete_a_values_8(
        partial_a,
        initial_guess=minerva_a_values[0:1] + minerva_a_values[5:],
        t0=minerva_t0,
    )
    FA_uni = F_A_z2(true_q2_values, a_uni, minerva_t0)
    MA_eff_uni = get_MA_effective(true_q2_values, FA_uni)
    w_uni = get_weight(MA_eff_uni, MA_grid, all_MA_weights)
    minerva_universe_event_weights.append(wc_weight_spline * w_uni)
print("Done.")

# --- PCA decomposition of the 4x4 MINERvA covariance matrix ---
# Eigenvalues/vectors sorted by descending eigenvalue (most significant first).
eigenvalues, eigenvectors = np.linalg.eigh(minerva_a_cov_matrix)
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]  # columns are eigenvectors

print("PCA eigenvalues (descending):", eigenvalues)

a_partial_cv = np.array(minerva_a_values[1:5])
pca_weights_hi = []
pca_weights_lo = []
for i in range(4):
    shift = np.sqrt(eigenvalues[i]) * eigenvectors[:, i]
    for sign, store in [(+1, pca_weights_hi), (-1, pca_weights_lo)]:
        a_partial = a_partial_cv + sign * shift
        a_full = complete_a_values_8(
            a_partial,
            initial_guess=minerva_a_values[0:1] + minerva_a_values[5:],
            t0=minerva_t0,
        )
        FA = F_A_z2(true_q2_values, a_full, minerva_t0)
        MA_eff = get_MA_effective(true_q2_values, FA)
        w = get_weight(MA_eff, MA_grid, all_MA_weights)
        store.append(wc_weight_spline * w)


def hist_band_from_universes(x_values, universe_event_weights, bins):
    """Return (lo, hi) histogram arrays from universe event weights (16th/84th percentile)."""
    uni_hists = np.array([
        np.histogram(x_values, bins=bins, weights=w)[0]
        for w in universe_event_weights
    ])
    return np.percentile(uni_hists, 16, axis=0), np.percentile(uni_hists, 84, axis=0)


def plot_with_bands(ax, bin_edges, n_cv, n_lo, n_hi, color, label):
    """Draw a step histogram + shaded uncertainty band."""
    ax.stairs(n_cv, bin_edges, color=color, label=label)
    ax.fill_between(
        np.repeat(bin_edges, 2)[1:-1],
        np.repeat(n_lo, 2),
        np.repeat(n_hi, 2),
        color=color, alpha=0.2,
    )


def frac_half_width(n_cv, n_lo, n_hi):
    """Return (n_hi - n_lo) / (2 * n_cv), with zeros where n_cv == 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_cv > 0, (n_hi - n_lo) / (2 * n_cv), 0.0)


PCA_COLORS = ["C2", "C3", "C4", "C5"]
MA_COLORS = plt.cm.RdYlBu_r(np.linspace(0.05, 0.95, 7))


def _ratio(n_num, n_denom):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_denom > 0, n_num / n_denom, np.nan)


def make_plot(x_values, bins, weights_default,
              all_dipole_w,
              minerva_w_cv, minerva_universe_w,
              pca_w_hi, pca_w_lo,
              axff_w,
              xlabel, xscale, filename):

    n_default, _ = np.histogram(x_values, bins=bins, weights=weights_default)
    # histograms for all 7 MA values; index 3 (MA=1.1) is the reference
    n_all_ma = [np.histogram(x_values, bins=bins, weights=w)[0] for w in all_dipole_w]
    n_dipole_cv = n_all_ma[3]   # MA=1.1
    n_dipole_lo = n_all_ma[2]   # MA=1.0
    n_dipole_hi = n_all_ma[4]   # MA=1.2
    n_minerva_cv, _ = np.histogram(x_values, bins=bins, weights=minerva_w_cv)
    n_minerva_lo, n_minerva_hi = hist_band_from_universes(x_values, minerva_universe_w, bins)

    # PCA component histograms
    n_pca_hi = [np.histogram(x_values, bins=bins, weights=w)[0] for w in pca_w_hi]
    n_pca_lo = [np.histogram(x_values, bins=bins, weights=w)[0] for w in pca_w_lo]

    # AxFFCCQEshape histograms (shape=0: dipole, shape=1: z-expansion)
    n_axff = [np.histogram(x_values, bins=bins, weights=w)[0] for w in axff_w]

    fig, (ax_main, ax_ratio, ax_axff, ax_frac) = plt.subplots(
        4, 1, figsize=(8, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 2], "hspace": 0.07},
    )

    # --- Main histogram panel ---
    ax_main.stairs(n_default, bins, color="black", label="Default weights")
    plot_with_bands(ax_main, bins, n_dipole_cv, n_dipole_lo, n_dipole_hi,
                    color="steelblue", label=r"Dipole $M_A = 1.1 \pm 0.1$ GeV")
    plot_with_bands(ax_main, bins, n_minerva_cv, n_minerva_lo, n_minerva_hi,
                    color="red", label="MINERvA z-expansion (±1σ)")
    ax_main.set_ylabel("Events")
    ax_main.legend()
    ax_main.set_title("True numuCC QE Events — Systematic Uncertainty Bands")
    ax_main.set_xlim(bins[0], bins[-1])
    if xscale == "log":
        ax_main.set_xscale("log")

    # --- MA ratio panel: everything / n at MA=1.1 ---
    for j, (n_ma, color) in enumerate(zip(n_all_ma, MA_COLORS)):
        ma_val = MA_grid[j]
        ax_ratio.stairs(_ratio(n_ma, n_dipole_cv), bins, color=color,
                        label=rf"$M_A={ma_val:.1f}$", linewidth=1.5)
    ax_ratio.stairs(_ratio(n_minerva_cv, n_dipole_cv), bins,
                    color="black", linewidth=1.5, label="MINERvA z-exp.")
    ax_ratio.axhline(1, color="black", linestyle=":", alpha=0.5)
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.legend(fontsize=6, ncol=4, loc="upper right")
    if xscale == "log":
        ax_ratio.set_xscale("log")

    # --- AxFFCCQEshape ratio panel: shape=0 and shape=1 / n at MA=1.1 ---
    ax_axff.stairs(_ratio(n_axff[0], n_dipole_cv), bins, color="C0", linewidth=1.5,
                   label="AxFFCCQEshape=0 (dipole)")
    ax_axff.stairs(_ratio(n_axff[1], n_dipole_cv), bins, color="C1", linewidth=1.5,
                   label="AxFFCCQEshape=1 (z-exp.)")
    ax_axff.axhline(1, color="black", linestyle=":", alpha=0.5)
    ax_axff.set_xlim(bins[0], bins[-1])
    ax_axff.set_ylim(0.5, 1.5)
    ax_axff.legend(fontsize=7, loc="upper right")
    if xscale == "log":
        ax_axff.set_xscale("log")

    # Shared y-axis label for the two ratio panels
    fig.canvas.draw()
    pos_ratio = ax_ratio.get_position()
    pos_axff = ax_axff.get_position()
    mid_y = (pos_ratio.y1 + pos_axff.y0) / 2
    fig.text(0.01, mid_y, r"Events / Events at $M_A=1.1$",
             va="center", ha="left", rotation="vertical", fontsize=9)

    # --- Fractional uncertainty panel ---
    frac_dipole = frac_half_width(n_dipole_cv, n_dipole_lo, n_dipole_hi)
    frac_minerva = frac_half_width(n_minerva_cv, n_minerva_lo, n_minerva_hi)

    # PCA components: each relative to minerva CV.
    # Use abs because a +1sigma shift in parameter space may decrease the histogram count,
    # so n_pca_hi - n_pca_lo can be negative; we want the magnitude.
    frac_pca = [np.abs(frac_half_width(n_minerva_cv, n_pca_lo[i], n_pca_hi[i])) for i in range(4)]
    frac_pca_quadsum = np.sqrt(sum(f**2 for f in frac_pca))

    # AxFFCCQEshape: uncertainty from difference between shape=0 and shape=1
    n_axff_lo = np.minimum(n_axff[0], n_axff[1])
    n_axff_hi = np.maximum(n_axff[0], n_axff[1])
    frac_axff = frac_half_width(n_dipole_cv, n_axff_lo, n_axff_hi)

    # Dipole + AxFFCCQEshape in quadrature
    frac_dipole_and_axff = np.sqrt(frac_dipole**2 + frac_axff**2)

    ax_frac.stairs(frac_dipole, bins, color="steelblue", linestyle="--",
                   label=r"Dipole $M_A = 1.1 \pm 0.1$ GeV")
    ax_frac.stairs(frac_minerva, bins, color="black",
                   label="MINERvA z-exp. (universes)")
    for i, (frac, color) in enumerate(zip(frac_pca, PCA_COLORS)):
        ax_frac.stairs(frac, bins, color=color, linestyle="--",
                       label=f"MINERvA PCA component {i+1}")
    ax_frac.stairs(frac_pca_quadsum, bins, color="red",
                   label="MINERvA z-exp. quad. sum")
    ax_frac.stairs(frac_axff, bins, color="purple", linestyle="--",
                   label="AxFFCCQEshape (0 vs 1)")
    ax_frac.stairs(frac_dipole_and_axff, bins, color="darkorange",
                   label=r"Dipole $\oplus$ AxFFCCQEshape")
    ax_frac.set_xlabel(xlabel)
    ax_frac.set_ylabel("Frac. unc.")
    ax_frac.set_ylim(0, 0.5)
    ax_frac.set_xlim(bins[0], bins[-1])
    ax_frac.legend(fontsize=7, loc="upper right")
    if xscale == "log":
        ax_frac.set_xscale("log")

    plt.savefig(filename, dpi=150)
    plt.close()


# ============================================================
# Plot 1: WC reco Enu
# ============================================================
make_plot(
    x_values=wc_energies,
    bins=np.linspace(0, 2500, 26),
    weights_default=weights,
    all_dipole_w=all_dipole_event_weights,
    minerva_w_cv=minerva_weights_cv, minerva_universe_w=minerva_universe_event_weights,
    pca_w_hi=pca_weights_hi, pca_w_lo=pca_weights_lo,
    axff_w=axff_event_weights,
    xlabel="WC kine_reco_Enu (MeV)", xscale="linear",
    filename="plots/uncertainty_wc_energy.png",
)

# ============================================================
# Plot 2: True Q²
# ============================================================
make_plot(
    x_values=true_q2_values,
    bins=np.logspace(-2, 2, 41),
    weights_default=weights,
    all_dipole_w=all_dipole_event_weights,
    minerva_w_cv=minerva_weights_cv, minerva_universe_w=minerva_universe_event_weights,
    pca_w_hi=pca_weights_hi, pca_w_lo=pca_weights_lo,
    axff_w=axff_event_weights,
    xlabel=r"True $Q^2$ (GeV$^2$)", xscale="log",
    filename="plots/uncertainty_true_q2.png",
)

print("Saved plots/uncertainty_wc_energy.png and plots/uncertainty_true_q2.png")

# ============================================================
# Plot 3: Reco muon energy in slices of muon angle
# ============================================================
cos_theta_slices = [
    (0.94, 1.00),
    (0.80, 0.94),
    (0.60, 0.80),
    (0.0, 0.60),
    (-1.0, 0.0),
]
slice_labels = [
    r"$\cos\theta_\mu \in [0.94, 1.00]$",
    r"$\cos\theta_\mu \in [0.80, 0.94)$",
    r"$\cos\theta_\mu \in [0.60, 0.80)$",
    r"$\cos\theta_\mu \in [0.0, 0.60)$",
    r"$\cos\theta_\mu < 0.0$",
]

n_slices = len(cos_theta_slices)
bins_mu_E = np.linspace(0, 3, 31)  # GeV

fig, axes = plt.subplots(
    3, n_slices, figsize=(4 * n_slices, 6),
    gridspec_kw={"height_ratios": [3, 1, 2], "hspace": 0.07, "wspace": 0.35},
)

for col, ((cos_lo, cos_hi), slabel) in enumerate(zip(cos_theta_slices, slice_labels)):
    mask = (cos_theta_mu >= cos_lo) & (cos_theta_mu < cos_hi)

    ax_main  = axes[0, col]
    ax_ratio = axes[1, col]
    ax_frac  = axes[2, col]
    ax_ratio.sharex(ax_main)
    ax_frac.sharex(ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_ratio.get_xticklabels(), visible=False)

    mu_E_slice = mu_E[mask]

    n_default_s, _ = np.histogram(mu_E_slice, bins=bins_mu_E, weights=weights[mask])
    n_all_ma_s = [np.histogram(mu_E_slice, bins=bins_mu_E, weights=w[mask])[0] for w in all_dipole_event_weights]
    n_dipole_cv_s = n_all_ma_s[3]
    n_dipole_lo_s = n_all_ma_s[2]
    n_dipole_hi_s = n_all_ma_s[4]
    n_minerva_cv_s, _ = np.histogram(mu_E_slice, bins=bins_mu_E, weights=minerva_weights_cv[mask])
    n_minerva_lo_s, n_minerva_hi_s = hist_band_from_universes(
        mu_E_slice, [w[mask] for w in minerva_universe_event_weights], bins_mu_E
    )
    n_pca_hi_s = [np.histogram(mu_E_slice, bins=bins_mu_E, weights=w[mask])[0] for w in pca_weights_hi]
    n_pca_lo_s = [np.histogram(mu_E_slice, bins=bins_mu_E, weights=w[mask])[0] for w in pca_weights_lo]

    # Main panel
    ax_main.stairs(n_default_s, bins_mu_E, color="black",
                   label="Default weights" if col == 0 else None)
    plot_with_bands(ax_main, bins_mu_E, n_dipole_cv_s, n_dipole_lo_s, n_dipole_hi_s,
                    color="steelblue",
                    label=r"Dipole $M_A = 1.1 \pm 0.1$ GeV" if col == 0 else None)
    plot_with_bands(ax_main, bins_mu_E, n_minerva_cv_s, n_minerva_lo_s, n_minerva_hi_s,
                    color="red",
                    label="MINERvA z-expansion (±1σ)" if col == 0 else None)
    ax_main.set_title(slabel, fontsize=9)
    ax_main.set_xlim(bins_mu_E[0], bins_mu_E[-1])
    if col == 0:
        ax_main.set_ylabel("Events")
        ax_main.legend(fontsize=7)

    # Ratio panel: everything / n at MA=1.1
    for j, (n_ma_s, color) in enumerate(zip(n_all_ma_s, MA_COLORS)):
        ax_ratio.stairs(_ratio(n_ma_s, n_dipole_cv_s), bins_mu_E, color=color,
                        label=rf"$M_A={MA_grid[j]:.1f}$" if col == 0 else None,
                        linewidth=1.5)
    ax_ratio.stairs(_ratio(n_minerva_cv_s, n_dipole_cv_s), bins_mu_E,
                    color="black", linewidth=1.5, label="MINERvA z-exp." if col == 0 else None)
    ax_ratio.axhline(1, color="black", linestyle=":", alpha=0.5)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlim(bins_mu_E[0], bins_mu_E[-1])
    if col == 0:
        ax_ratio.set_ylabel(r"Events / Events at $M_A=1.1$")
        ax_ratio.legend(fontsize=5, ncol=4, loc="upper right")

    # Fractional uncertainty panel
    frac_dipole_s = frac_half_width(n_dipole_cv_s, n_dipole_lo_s, n_dipole_hi_s)
    frac_minerva_s = frac_half_width(n_minerva_cv_s, n_minerva_lo_s, n_minerva_hi_s)
    frac_pca_s = [np.abs(frac_half_width(n_minerva_cv_s, n_pca_lo_s[i], n_pca_hi_s[i])) for i in range(4)]
    frac_pca_quadsum_s = np.sqrt(sum(f**2 for f in frac_pca_s))

    ax_frac.stairs(frac_dipole_s, bins_mu_E, color="steelblue",
                   label=r"Dipole $M_A = 1.1 \pm 0.1$ GeV" if col == 0 else None)
    ax_frac.stairs(frac_minerva_s, bins_mu_E, color="black",
                   label="MINERvA z-exp. (universes)" if col == 0 else None)
    for i, (frac_s, c) in enumerate(zip(frac_pca_s, PCA_COLORS)):
        ax_frac.stairs(frac_s, bins_mu_E, color=c,
                       label=f"MINERvA PCA component {i+1}" if col == 0 else None)
    ax_frac.stairs(frac_pca_quadsum_s, bins_mu_E, color="red", linestyle="--",
                   label="MINERvA z-exp. quad. sum" if col == 0 else None)
    ax_frac.set_ylim(0, 0.3)
    ax_frac.set_xlim(bins_mu_E[0], bins_mu_E[-1])
    ax_frac.set_xlabel("Reco muon energy (GeV)")
    if col == 0:
        ax_frac.set_ylabel("Frac. unc.")
        ax_frac.legend(fontsize=6)

fig.suptitle("True numuCC QE Events — Reco muon energy in muon angle slices", fontsize=11)
plt.savefig("plots/uncertainty_muon_energy_angle_slices.png", dpi=150)
plt.close()

print("Saved plots/uncertainty_muon_energy_angle_slices.png")
