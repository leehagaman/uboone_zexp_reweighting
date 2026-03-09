import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from axial_form_factor_parametrizations import (
    F_A_z2, F_A_dipole,
    minerva_a_values, minerva_t0, minerva_a_universes,
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
