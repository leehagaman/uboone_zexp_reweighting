import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from axial_form_factor_parametrizations import (
    F_A_z2, get_MA_effective,
    minerva_a_values, minerva_t0,
)

os.makedirs("plots", exist_ok=True)

ROOT_FILE = "/nevis/riverside/data/leehagaman/ngem/data_files/nu_overlay_splines_50.root"
EVENT_INDEX = 4

MA_GRID = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])

f = uproot.open(ROOT_FILE)

true_q2_all   = f["singlephotonana"]["eventweight_tree"]["GTruth_gQ2"].array(library="np")
run_all       = f["singlephotonana"]["eventweight_tree"]["run"].array(library="np")
subrun_all    = f["singlephotonana"]["eventweight_tree"]["subrun"].array(library="np")
event_all     = f["singlephotonana"]["eventweight_tree"]["event"].array(library="np")
ma_spline_raw = f["spline_weights"]["MaCCQE_UBGenie"].array(library="np")

q2        = float(true_q2_all[EVENT_INDEX])
run       = int(run_all[EVENT_INDEX])
subrun    = int(subrun_all[EVENT_INDEX])
event     = int(event_all[EVENT_INDEX])
weights_7 = np.array([float(ma_spline_raw[EVENT_INDEX][j]) for j in range(7)])
weights_7 = np.where(np.isfinite(weights_7), weights_7, 1.0)

fa_val = float(F_A_z2(np.array([q2]), minerva_a_values, minerva_t0)[0])
ma_eff = float(get_MA_effective(np.array([q2]), np.array([fa_val]))[0])

print(f"Event {EVENT_INDEX}")
print(f"  Q^2            = {q2:.4f} GeV^2")
print(f"  F_A (MINERvA)  = {fa_val:.4f}")
print(f"  M_A_eff        = {ma_eff:.4f} GeV")
print(f"  Weights at grid: {weights_7}")

# Linear interpolation (same logic as get_weight)
idx_lo = int(np.clip(np.searchsorted(MA_GRID, ma_eff, side='right') - 1, 0, len(MA_GRID) - 2))
idx_hi = idx_lo + 1
t = (ma_eff - MA_GRID[idx_lo]) / (MA_GRID[idx_hi] - MA_GRID[idx_lo])
interp_weight = weights_7[idx_lo] * (1 - t) + weights_7[idx_hi] * t
print(f"  Bracket: MA_grid[{idx_lo}]={MA_GRID[idx_lo]}, MA_grid[{idx_hi}]={MA_GRID[idx_hi]}, t={t:.3f}")
print(f"  Interpolated weight = {interp_weight:.4f}")

# Dense curve for the full piecewise-linear shape
margin = 0.15
ma_dense = np.linspace(MA_GRID[0] - margin, MA_GRID[-1] + margin, 500)
w_dense = np.empty_like(ma_dense)
for k, m in enumerate(ma_dense):
    i_lo = int(np.clip(np.searchsorted(MA_GRID, m, side='right') - 1, 0, len(MA_GRID) - 2))
    i_hi = i_lo + 1
    tk = (m - MA_GRID[i_lo]) / (MA_GRID[i_hi] - MA_GRID[i_lo])
    w_dense[k] = weights_7[i_lo] * (1 - tk) + weights_7[i_hi] * tk

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(ma_dense, w_dense, color='steelblue', lw=1.5, zorder=1,
        label='Piecewise-linear interpolation')
ax.scatter(MA_GRID, weights_7, color='steelblue', s=60, zorder=3,
           label='Weights at $M_A$ grid points')
ax.scatter([MA_GRID[idx_lo], MA_GRID[idx_hi]],
           [weights_7[idx_lo], weights_7[idx_hi]],
           color='darkorange', s=100, zorder=4,
           label=f'Bracket points (indices {idx_lo}, {idx_hi})')
ax.axvline(ma_eff, color='crimson', lw=1.2, ls='--', zorder=2)
ax.axhline(interp_weight, color='crimson', lw=1.2, ls='--', zorder=2)
ax.scatter([ma_eff], [interp_weight], color='crimson', s=120, zorder=5, marker='*',
           label=f'$M_{{A,\\mathrm{{eff}}}}={ma_eff:.3f}$ GeV → weight={interp_weight:.4f}')
ax.axhline(1.0, color='gray', lw=0.8, ls=':', zorder=0, label='weight = 1')
ax.text(0.02, 0.97,
        f'$Q^2 = {q2:.3f}$ GeV$^2$\n'
        f'$F_A$(MINERvA CV) $= {fa_val:.4f}$\n'
        f'$M_{{A,\\mathrm{{eff}}}} = {ma_eff:.4f}$ GeV',
        transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

ax.set_xlabel('$M_A^{\\mathrm{eff}}$ [GeV]', fontsize=12)
ax.set_ylabel('Event weight', fontsize=12)
filename = os.path.basename(ROOT_FILE)
ax.set_title(f'{filename}  |  run {run}, subrun {subrun}, event {event}', fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"plots/event_reweighting_{EVENT_INDEX}.png", dpi=150)
plt.close()

print(f"Saved plots/event_reweighting_{EVENT_INDEX}.png")
