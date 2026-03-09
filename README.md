# MicroBooNE Z-expansion Axial Form Factor Reweighting

Intended to be an alternative axial form factor treatment for the [MicroBooNE tune GENIE model](https://doi.org/10.1103/PhysRevD.105.072001), which currently uses an axial mass of M_A = 1.1 +/- 0.1 GeV.

With newly generated spline systematics giving event weights for 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, and 1.4 GeV M_A values, we can extrapolate between these values event-by-event to get weights for a broad range of axial form factor values.

Currently, the plan is to reweight to the MINERvA z-expansion axial form factor extracted values, which would match what [Nathaniel Rowe is currently working on for SBN uncertainties](https://sbn-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=44211). Then, using PROfit, we could extract best-fit values and uncertainties for these z-expansion parameters (defined using a PCA on the MINERvA covariance matrix) using MicroBooNE data.
