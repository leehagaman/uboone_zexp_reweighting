import numpy as np
from scipy.optimize import fsolve


pion_mass = 0.139570 # GeV/c^2
t_cut = 9 * pion_mass * pion_mass

def F_A_z2(q2, a_values, t0):

    z = (np.sqrt(t_cut + q2) - np.sqrt(t_cut - t0)) / (np.sqrt(t_cut + q2) + np.sqrt(t_cut - t0))

    ret = 0
    for i in range(len(a_values)):
        ret += a_values[i] * z**i

    return ret

def F_A_z2_func_z(z, a_values, t0):

    ret = 0
    for i in range(len(a_values)):
        ret += a_values[i] * z**i

    return ret


def F_A_dipole(q2, M_A):

    F_A_0 = -1.2723 # ± 0.0023

    return F_A_0 / (1 + q2 / (M_A**2))**2


def get_MA_effective(q2, F_A):
    # only accurate at a specific Q^2 value, useful for reweighting a specific event with known weights for different M_A values

    # effective MA such that F_A_dipole(Q^2, MA_eff) = F_A_minerva(Q^2).
    # F_A_0 / (1 + Q^2 / (MA_eff^2))**2 = F_A_minerva
    # (1 + Q^2 / (MA_eff^2))**2 = F_A_0 / F_A_minerva
    # (1 + Q^2 / (MA_eff^2)) = sqrt(F_A_0 / F_A_minerva)
    # Q^2 / (MA_eff^2) = sqrt(F_A_0 / F_A_minerva) - 1
    # MA_eff^2 = Q^2 / (sqrt(F_A_0 / F_A_minerva) - 1)
    # MA_eff = sqrt(Q^2 / (sqrt(F_A_0 / F_A_minerva) - 1))

    F_A_0 = -1.2723
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = F_A_0 / F_A  # positive (both negative)
        denom = np.sqrt(ratio) - 1
        MA_eff = np.where(
            (ratio > 0) & (denom > 1e-10),
            np.sqrt(q2 / denom),
            np.nan,
        )

    return MA_eff


def complete_a_values_8(a_values, t0, initial_guess):

    # parameters to be solved: a0, a5, a6, a7, a8,
    # equations: F(0) = -1.2723, four sum rules

    z_at_Q2_0 = (np.sqrt(t_cut + 0) - np.sqrt(t_cut - t0)) / (np.sqrt(t_cut + 0) + np.sqrt(t_cut - t0))

    a1, a2, a3, a4 = a_values

    def equations(x):
        
        F0_equation = (x[0] * z_at_Q2_0**0
                        + a1 * z_at_Q2_0**1
                        + a2 * z_at_Q2_0**2
                        + a3 * z_at_Q2_0**3
                        + a4 * z_at_Q2_0**4
                        + x[1] * z_at_Q2_0**5
                        + x[2] * z_at_Q2_0**6
                        + x[3] * z_at_Q2_0**7
                        + x[4] * z_at_Q2_0**8
                         + 1.2723)
        
        # sum from n to infinity of k(k-1)...(k-n+1) a_k = 0
        # sum from n to 8 of k(k-1)...(k-n+1) a_k = 0

        # n = 0, sum from 0 to 8 of k(k-1)...(k+1) a_k = 0
        # I guess that means sum from 0 to 8 of a_k = 0
        sum_rule_equation_0 = (x[0] + a1 + a2 + a3 + a4 + x[1] + x[2] + x[3] + x[4])

        # n = 1, sum from 1 to 8 of k(k-1)...(k) a_k = 0
        # I guess that means sum from 1 to 8 of k * a_k = 0
        sum_rule_equation_1 = (
              a1 * 1
            + a2 * 2
            + a3 * 3
            + a4 * 4
            + x[1] * 5
            + x[2] * 6
            + x[3] * 7
            + x[4] * 8
        )

        # n = 2, sum from 2 to 8 of k(k-1)...(k-1) a_k = 0
        # I guess that means sum from 2 to 8 of k(k-1) * a_k = 0
        sum_rule_equation_2 = (
              a2 * 2 * 1
            + a3 * 3 * 2
            + a4 * 4 * 3
            + x[1] * 5 * 4
            + x[2] * 6 * 5
            + x[3] * 7 * 6
            + x[4] * 8 * 7
        )

        # n = 3, sum from 3 to 8 of k(k-1)...(k-1) a_k = 0
        # I guess that means sum from 3 to 8 of k(k-1)(k-2) * a_k = 0
        sum_rule_equation_3 = (
              a3 * 3 * 2 * 1
            + a4 * 4 * 3 * 2
            + x[1] * 5 * 4 * 3
            + x[2] * 6 * 5 * 4
            + x[3] * 7 * 6 * 5
            + x[4] * 8 * 7 * 6
        )

        return np.array([F0_equation, sum_rule_equation_0, sum_rule_equation_1, sum_rule_equation_2, sum_rule_equation_3])
    
    result = list(fsolve(equations, initial_guess))

    #print("result", result)
    #print([result[0]], a_values, result[1:])
    #print("ret", [result[0]] + a_values + result[1:])
    
    return [result[0]] + list(a_values) + result[1:]


# taken from https://www.nature.com/articles/s41586-022-05478-3#Sec18 supplementary information, supplementary table 4

minerva_t0 = -0.75 # GeV/c^2
minerva_a_values = [-0.50, 1.50, -1.2, -0.1, 0.2, 0.46, -0.40, 0.15, -0.044]

minerva_a_errors = [0.31, 0.7, 1.9, 3.5]
minerva_a_corr_matrix = np.array([
    [1., 0.012, -0.93, 0.52],
    [0.012, 1., -0.32, -0.78],
    [-0.93, -0.32, 1., -0.27],
    [0.52, -0.78, -0.27, 1.],
])

minerva_a_cov_matrix = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        minerva_a_cov_matrix[i, j] = minerva_a_corr_matrix[i, j] * minerva_a_errors[i] * minerva_a_errors[j]

recalc_minerva_a_values = complete_a_values_8(minerva_a_values[1:5], initial_guess=minerva_a_values[0:1]+minerva_a_values[5:], t0=minerva_t0)

minerva_partial_a_universes = np.random.multivariate_normal(minerva_a_values[1:5], minerva_a_cov_matrix, 1000)
minerva_a_universes = []
for uni_i in range(len(minerva_partial_a_universes)):
    minerva_a_universes.append(complete_a_values_8(minerva_partial_a_universes[uni_i], initial_guess=minerva_a_values[0:1]+minerva_a_values[5:], t0=minerva_t0))




# This is just Q^2 < 1 GeV^2, from https://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.113015

deuterium_partial_a_values = [2.3, -0.6, -3.8, 2.3]
deuterium_a_errors = np.sqrt(np.array([0.0154, 1.08, 6.54, 7.40]))
deuterium_a_corr_matrix = np.array([
    [1, 0.335, -0.678, 0.611],
    [0.350, 1, -0.898, 0.367],
    [-0.678, -0.898, 1, -0.685],
    [0.611, 0.367, -0.685, 1],
])

deuterium_t0 = -0.28 # GeV/c^2

deuterium_a_cov_matrix = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        deuterium_a_cov_matrix[i, j] = deuterium_a_corr_matrix[i, j] * deuterium_a_errors[i] * deuterium_a_errors[j]

deuterium_a_values = complete_a_values_8(deuterium_partial_a_values, initial_guess=[1,1,1,1,1], t0=deuterium_t0)



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
