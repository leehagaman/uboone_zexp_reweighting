# MicroBooNE Z-expansion Axial Form Factor Reweighting

Intended to be an alternative axial form factor treatment for the [MicroBooNE tune GENIE model](https://doi.org/10.1103/PhysRevD.105.072001), which currently uses an axial mass of M_A = 1.1 +/- 0.1 GeV.

With newly generated spline systematics giving event weights for 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, and 1.4 GeV M_A values, we can extrapolate between these values event-by-event to get weights for a broad range of axial form factor values.

Currently, the plan is to reweight to the MINERvA z-expansion axial form factor extracted values, which would match what [Nathaniel Rowe is currently working on for SBN uncertainties](https://sbn-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=44211). Then, using PROfit, we could extract best-fit values and uncertainties for these z-expansion parameters (defined using a PCA on the MINERvA covariance matrix) using MicroBooNE data.

## Applying the Weights

After running `write_fa_weights_to_root.py`, three kinds of weights are available in the `spline_weights` tree.

### 1. Default dipole M_A CV (MA = 1.1 GeV)

This is the standard MicroBooNE tune central value. No new branches are needed; use the existing spline weight at the MA = 1.1 index (index 3 of the 7-point grid 0.8–1.4 GeV):

```
event_weight = weight_spline * weight_cv # equivalent to weight_spline * MaCCQE_UBGenie[3]
```

### 2. MINERvA z-expansion CV

`weight_minerva_FA` is the z-expansion reweighting factor, analogous to `MaCCQE_UBGenie[3]`. Multiply by `weight_spline` to get the full event weight:

```
event_weight = weight_spline * weight_minerva_FA
```

Internally this is computed by:

1. Evaluating `F_A(Q²)` with the MINERvA z-expansion parameters.
2. Finding the effective dipole mass `M_A^eff(Q²)` such that `F_A^dipole(Q², M_A^eff) = F_A^zexp(Q²)`.
3. Interpolating the event's 7-point MA spline weight at `M_A^eff`.
4. Multiplying by `weight_spline`.

### 3. Systematic variations (PCA components)

`weight_spline_FAzexpPCA{1-4}` encode ±3σ excursions along the four dominant PCA eigenvectors of the MINERvA covariance matrix. Each branch is a 7-element array per event, corresponding to σ = −3, −2, −1, 0, +1, +2, +3.

**Using discrete σ steps:**

```
# e.g., PCA component 1 at +1σ  (sigma index 4 in the array)
event_weight = weight_spline * weight_spline_FAzexpPCA1[4]
```

The σ = 0 entry (index 3) equals `weight_minerva_FA` for every event.

## Scripts

```
# Makes a comparison of different F_A vs Q^2 curves
python plot_F_A.py

# Makes plots comparing different F_A treatments in true Q^2 as well as reco space
# Done independently of any root file saving or loading
python zexp_uncertainties.py

# Adds new F_A spline weights to a root file
python write_fa_weights_to_root.py /nevis/riverside/data/leehagaman/ngem/data_files/nu_overlay_splines_50.root;
python write_fa_weights_to_root.py /nevis/riverside/data/leehagaman/ngem/data_files/run4b_dirt_surprise_200_splines.root;
python write_fa_weights_to_root.py /nevis/riverside/data/leehagaman/ngem/data_files/run4b_nuoverlay_retuple_splines.root;

# Loads new F_A spline weights from a root file and makes some plots to visualize
python plot_from_rootfile.py

```
