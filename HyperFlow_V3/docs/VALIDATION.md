# Verification

HyperFlow-2D is demonstrated with two reacting-flow benchmarks:

- Planar CJ detonation: pressure profiles compared with Sun et al. (2023), Section 3.5 and Fig. 8(a).
- Wedge-induced oblique detonation: temperature fields and pressure contours compared with Teng et al. (2021), Case 1 and Fig. 3(a).

The README contains the two selected comparison figures and their publication references.

The solver checks cover boundary conditions, conservative mesh refinement, species closure, constant-volume chemical integration and parallel consistency. Both case configurations use the same HyperFlow-2D executable.
