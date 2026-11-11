# Notices

This snapshot does not introduce a project-level license grant for the HyperFlow source. Dependency licenses remain with their respective authors.

- AMReX: fixed commit `2d8c2e7e9d782cee6a040c5a503497c4c0708c96`; BSD-3-Clause. Its license is reproduced in `docs/AMReX-LICENSE`. AMReX is obtained separately by CMake and is not bundled as a source tree.
- Cantera: BSD-3-Clause; linked as an external dependency.
- SUNDIALS: BSD-3-Clause; linked as an external dependency.
- Eigen: MPL-2.0; external headers.
- fmt and nlohmann/json: MIT; external library/headers.

`H2_Li.yaml` is the unmodified local benchmark mechanism. Its original metadata states `ctml2yaml`, Cantera 3.0.0, input `chem.xml`, conversion date 29 September 2023. The file has nine species, 21 reactions and NASA7 thermochemistry. The exact upstream provenance and a separate mechanism license were not established by this packaging; no equivalence with every published Li-mechanism variant is implied. The exact file hash is preserved in the provenance record.

Benchmark publications are cited in the README. Digitized values are approximate figure readings; the original articles, panels and reference-solver code are not redistributed in this snapshot.
