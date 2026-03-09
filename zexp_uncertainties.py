
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

# --- M_A = 1.1 ± 0.1 dipole weights ---
# lo: MA=1.0 (index 2), cv: MA=1.1 (index 3), hi: MA=1.2 (index 4)
MA_1p0_weights = all_MA_weights[:, 2]
MA_1p1_weights = all_MA_weights[:, 3]
MA_1p2_weights = all_MA_weights[:, 4]

dipole_weights_cv = wc_weight_spline * MA_1p1_weights
dipole_weights_lo = wc_weight_spline * MA_1p0_weights
dipole_weights_hi = wc_weight_spline * MA_1p2_weights

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


def make_plot(x_values, bins, weights_default,
              dipole_w_cv, dipole_w_lo, dipole_w_hi,
              minerva_w_cv, minerva_universe_w,
              pca_w_hi, pca_w_lo,
              xlabel, xscale, filename):

    n_default, _ = np.histogram(x_values, bins=bins, weights=weights_default)
    n_dipole_cv, _ = np.histogram(x_values, bins=bins, weights=dipole_w_cv)
    n_dipole_lo, _ = np.histogram(x_values, bins=bins, weights=dipole_w_lo)
    n_dipole_hi, _ = np.histogram(x_values, bins=bins, weights=dipole_w_hi)
    n_minerva_cv, _ = np.histogram(x_values, bins=bins, weights=minerva_w_cv)
    n_minerva_lo, n_minerva_hi = hist_band_from_universes(x_values, minerva_universe_w, bins)

    # PCA component histograms
    n_pca_hi = [np.histogram(x_values, bins=bins, weights=w)[0] for w in pca_w_hi]
    n_pca_lo = [np.histogram(x_values, bins=bins, weights=w)[0] for w in pca_w_lo]

    fig, (ax_main, ax_ratio, ax_frac) = plt.subplots(
        3, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 2], "hspace": 0.07},
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

    # --- Ratio panel: MINERvA CV / dipole CV ---
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(n_dipole_cv > 0, n_minerva_cv / n_dipole_cv, np.nan)
    ax_ratio.stairs(ratio, bins, color="black")
    ax_ratio.axhline(1, color="black", linestyle=":", alpha=0.5)
    ax_ratio.set_ylabel("MINERvA / Dipole")
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.set_ylim(0.5, 1.5)
    if xscale == "log":
        ax_ratio.set_xscale("log")

    # --- Fractional uncertainty panel ---
    frac_dipole = frac_half_width(n_dipole_cv, n_dipole_lo, n_dipole_hi)
    frac_minerva = frac_half_width(n_minerva_cv, n_minerva_lo, n_minerva_hi)

    # PCA components: each relative to minerva CV.
    # Use abs because a +1sigma shift in parameter space may decrease the histogram count,
    # so n_pca_hi - n_pca_lo can be negative; we want the magnitude.
    frac_pca = [np.abs(frac_half_width(n_minerva_cv, n_pca_lo[i], n_pca_hi[i])) for i in range(4)]
    frac_pca_quadsum = np.sqrt(sum(f**2 for f in frac_pca))

    """print(f"\n--- Fractional uncertainties per bin [{filename}] ---")
    print(f"{'Bin':>5}  {'lo edge':>10}  {'hi edge':>10}  {'dipole':>8}  {'minerva':>8}  " +
          "  ".join(f"{'PCA'+str(i+1):>8}" for i in range(4)) + "  {'quadsum':>8}")
    for b in range(len(bins) - 1):
        pca_vals = "  ".join(f"{frac_pca[i][b]:8.4f}" for i in range(4))
        print(f"{b:>5}  {bins[b]:10.4g}  {bins[b+1]:10.4g}  {frac_dipole[b]:8.4f}  "
              f"{frac_minerva[b]:8.4f}  {pca_vals}  {frac_pca_quadsum[b]:8.4f}")
    """

    ax_frac.stairs(frac_dipole, bins, color="steelblue", label=r"Dipole $M_A = 1.1 \pm 0.1$ GeV")
    ax_frac.stairs(frac_minerva, bins, color="black", label="MINERvA z-exp. (universes)")
    for i, (frac, color) in enumerate(zip(frac_pca, PCA_COLORS)):
        ax_frac.stairs(frac, bins, color=color, label=f"MINERvA PCA component {i+1}")
    ax_frac.stairs(frac_pca_quadsum, bins, color="red", linestyle="--",
                   label="MINERvA z-exp. quad. sum")
    ax_frac.set_xlabel(xlabel)
    ax_frac.set_ylabel("Frac. unc.")
    ax_frac.set_ylim(0, 0.3)
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
    dipole_w_cv=dipole_weights_cv, dipole_w_lo=dipole_weights_lo, dipole_w_hi=dipole_weights_hi,
    minerva_w_cv=minerva_weights_cv, minerva_universe_w=minerva_universe_event_weights,
    pca_w_hi=pca_weights_hi, pca_w_lo=pca_weights_lo,
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
    dipole_w_cv=dipole_weights_cv, dipole_w_lo=dipole_weights_lo, dipole_w_hi=dipole_weights_hi,
    minerva_w_cv=minerva_weights_cv, minerva_universe_w=minerva_universe_event_weights,
    pca_w_hi=pca_weights_hi, pca_w_lo=pca_weights_lo,
    xlabel=r"True $Q^2$ (GeV$^2$)", xscale="log",
    filename="plots/uncertainty_true_q2.png",
)

print("Saved plots/uncertainty_wc_energy.png and plots/uncertainty_true_q2.png")
