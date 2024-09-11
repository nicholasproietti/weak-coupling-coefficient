# weak_coupling_coefficient

This respostory contains two codes which calculate the weak coupling coefficient *q* for mixed-mode oscillations within red giant branch models computed by the [MESA stellar evolution code](https://docs.mesastar.org/en/24.08.1/) and [GYRE stellar oscillation code](https://gyre.readthedocs.io/en/stable/). These codes are parts of my project as a summer research student at the Heidelberg Institute for Theoretical Studies (HITS gGmbH) in Summer 2024.

`propagation_diagram` - Illustrates the propagation diagram of an RGB model and calculates the weak coupling coefficient `q` for several mixed-modes using the asymptotic expression defined in [Shibahashi (1979)](https://ui.adsabs.harvard.edu/abs/1979PASJ...31...87S/abstract).

`coupling_coeff_analysis` - For a given RGB model, calculates the weak coupling coefficient `q` by fitting the asymptotic solutions of Eq. 9 of [Mosser et al. (2012a)] to the GYRE-computed frequencies.
