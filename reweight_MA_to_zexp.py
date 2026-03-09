
import uproot
import numpy as np
import matplotlib.pyplot as plt
from z_expansion_functions_values import F_A_z2, F_A_dipole, get_MA_effective, minerva_a_values, minerva_t0


def get_weight(MA_eff, MA_grid, all_MA_weights):
    """
    For each event, interpolate (or linearly extrapolate) its reweighting factor
    at MA_eff from the known weights at the discrete MA_grid points.

    MA_eff         : (N,) effective MA per event, may contain nan
    MA_grid        : (K,) sorted array of MA values at which weights are known
    all_MA_weights : (N, K) event weights at each MA_grid point

    Returns (N,) weight array; entries where MA_eff is nan (inversion undefined)
    or where the result is otherwise invalid are set to 1.
    """
    # Replace nan MA_eff with a safe sentinel so searchsorted doesn't misbehave;
    # those entries will be overwritten to 1 at the end.
    MA_safe = np.where(np.isnan(MA_eff), MA_grid[0], MA_eff)

    # Find the index of the left edge of the interval containing each MA_eff.
    # searchsorted(..., side='right') - 1 gives the left neighbor index.
    idx_lo = np.clip(
        np.searchsorted(MA_grid, MA_safe, side='right') - 1,
        0, len(MA_grid) - 2,
    )
    idx_hi = idx_lo + 1

    # Fractional position within the interval [MA_grid[idx_lo], MA_grid[idx_hi]].
    # t < 0  =>  MA_eff is below the grid  =>  linear extrapolation to the left.
    # t > 1  =>  MA_eff is above the grid  =>  linear extrapolation to the right.
    t = (MA_safe - MA_grid[idx_lo]) / (MA_grid[idx_hi] - MA_grid[idx_lo])

    # Linear blend: w_lo*(1-t) + w_hi*t
    arange = np.arange(len(MA_eff))
    result = all_MA_weights[arange, idx_lo] * (1 - t) + all_MA_weights[arange, idx_hi] * t

    # Revert any undefined or nonsensical entries to 1 (no reweighting).
    bad = np.isnan(MA_eff) | np.isnan(result) | np.isinf(result) | (result < 0)
    result[bad] = 1
    return result


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

wc_energies = f["wcpselection"]["T_KINEvars"].arrays("kine_reco_Enu", library="np")["kine_reco_Enu"][true_numuCCQE_flags]

wc_weight_cv = f["wcpselection"]["T_eval"].arrays(["weight_cv"], library="np")["weight_cv"][true_numuCCQE_flags]
wc_weight_spline = f["wcpselection"]["T_eval"].arrays(["weight_spline"], library="np")["weight_spline"][true_numuCCQE_flags]

weights = wc_weight_cv * wc_weight_spline
weights[weights < 0] = 1
weights[weights > 30] = 1
weights[np.isnan(weights)] = 1
weights[np.isinf(weights)] = 1

MA_spline_weights = f["spline_weights"].arrays(["MaCCQE_UBGenie"], library="np")["MaCCQE_UBGenie"]

MA_1p1_weights = np.array([x[3] for x in MA_spline_weights])[true_numuCCQE_flags]
MA_1p3_weights = np.array([x[5] for x in MA_spline_weights])[true_numuCCQE_flags]

MA_1p1_weights[np.isnan(MA_1p1_weights)] = 1
MA_1p1_weights[np.isinf(MA_1p1_weights)] = 1
MA_1p3_weights[np.isnan(MA_1p3_weights)] = 1
MA_1p3_weights[np.isinf(MA_1p3_weights)] = 1

# MINERvA z-expansion reweighting
# Load all 7 MA weights: indices 0-6 correspond to MA = 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4
MA_grid = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
all_MA_weights = np.column_stack([
    np.array([x[j] for x in MA_spline_weights])[true_numuCCQE_flags]
    for j in range(7)
])  # shape (N_events, 7)
for j in range(7):
    col = all_MA_weights[:, j]
    col[np.isnan(col)] = 1
    col[np.isinf(col)] = 1

# For each event, find the effective MA such that F_A_dipole(Q^2, MA_eff) = F_A_minerva(Q^2).
FA_minerva_per_event = F_A_z2(true_q2_values, minerva_a_values, minerva_t0)
MA_eff = get_MA_effective(true_q2_values, FA_minerva_per_event)

minerva_zexp_weights = get_weight(MA_eff, MA_grid, all_MA_weights)

bins = np.linspace(0, 2500, 26)

plt.figure(figsize=(8, 6))
n, bins, patches = plt.hist(wc_energies, bins=bins, weights=weights, histtype="step", label="default weights")
#n_1p1, bins, patches = plt.hist(wc_energies, bins=bins, weights=wc_weight_spline * MA_1p1_weights, histtype="step", label=r"$M_A=1.1$ GeV weights")
n_1p3, bins, patches = plt.hist(wc_energies, bins=bins, weights=wc_weight_spline * MA_1p3_weights, histtype="step", label=r"$M_A=1.3$ GeV weights")
plt.hist(wc_energies, bins=bins, weights=wc_weight_spline * minerva_zexp_weights, histtype="step", label="MINERvA z-expansion weights")
plt.xlabel("WC kine_reco_Enu (MeV)")
plt.ylabel("Events")
plt.legend()
plt.title("True numuCC QE Events")
plt.savefig("plots/wc_energy_hist.png")
plt.close()

bins = np.logspace(-2, np.log10(10), 26)
plt.figure(figsize=(8, 6))
n, bins, patches = plt.hist(true_q2_values, bins=bins, weights=weights, histtype="step", label="default weights")
#n_1p1, bins, patches = plt.hist(wc_energies, bins=bins, weights=wc_weight_spline * MA_1p1_weights, histtype="step", label=r"$M_A=1.1$ GeV weights")
n_1p3, bins, patches = plt.hist(true_q2_values, bins=bins, weights=wc_weight_spline * MA_1p3_weights, histtype="step", label=r"$M_A=1.3$ GeV weights")
plt.hist(true_q2_values, bins=bins, weights=wc_weight_spline * minerva_zexp_weights, histtype="step", label="MINERvA z-expansion weights")
plt.xscale("log")
plt.xlabel("True Q2 (GeV^2)")
plt.ylabel("Events")
plt.legend()
plt.title("True numuCC QE Events")
plt.savefig("plots/true_q2_hist.png")
plt.close()
