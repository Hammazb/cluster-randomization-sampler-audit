# Code and data: Certifying approximate samplers for constrained cluster randomization

Authors: Muhammad Hammaz Buksh, Muhammad Faisal Sindhi (University of South Florida).

Reproduces Sections 6, 7.1, 7.2, 7.4 and Appendix B of the paper. Python 3, numpy, scipy (qiskit only for hardware/ circuit prep).

## Run order (from `sim/`)
1. `python3 target.py` – exact accepted set (1,287 of 12,870 allocations) -> `target.npz`, `target_summary.json`
2. `python3 classical.py` – swap chain (writes `swapP.npy`, 13 MB), biased descent, `classical_summary.json`
3. `python3 run_xy.py`, `python3 run_pen.py`, `python3 run_ext.py` – XY-QAOA, penalty-QAOA, Grover-mixer/ceiling runs (saved outputs included: `xy_runs.json`, `pen_runs.json`, `ext_runs.json`)
4. `python3 run_infer.py 50000` – inference study, 50,000 replicates per cell (~2 min). Output `infer_results.json` is bit-identical to the included `infer_results_R50000.json` (verified by clean re-run).

## Data
`dickinson_design.csv` and `cvc/`: pre-randomization county-level covariates of the Dickinson et al. (2015) example, as distributed in the cvcrand R package. No outcomes or individual-level data are included.

## Hardware (`hardware/`)
8-cluster instance run on ibm_fez (4,096 shots per depth, no error mitigation). `small_hw.py` builds/trains circuits; `xy_p*.qasm` are the transpiled circuits; `hw_counts_raw.json` holds raw counts and job ids; `hw_summary.json` the metrics. Not covered: Section 7.3 (Tables 4-5) and Appendix A come from the original frozen run and are not in this bundle.

## Integrity
`SHA256SUMS` lists SHA-256 hashes of every file.

## License
Code: MIT (see LICENSE). Data: as distributed by cvcrand (GPL-2); cite Dickinson et al. (2015).
