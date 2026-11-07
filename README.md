# Causal discovery with false positive error control

Research code for m-conservative PC (mCPC), by Jahn, Chandrasekaran,
Eberhardt, and Schulman. MIT license. Experiments require macOS/Linux;
Python 3.14.3 was used for the paper.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt -e .
python -m unittest discover -s tests
```

## Use mCPC

```python
import numpy as np
from mcpc import mcpc

# X: finite continuous data, rows = samples, columns = variables.
X = np.asarray(X, dtype=float)
X = (X - X.mean(axis=0)) / X.std(axis=0)  # remove constant columns first
result = mcpc(X, m=1, alpha=0.1, ugl_penalty=0.01)
result.pattern          # .directed, .undirected, .ambiguous
result.representatives  # one networkx.DiGraph per represented equivalence class
result.diagnostics     # test counts, threshold, learned neighborhoods
```

Vertices are column indices; ambiguous triples are `(i, center, j)`, with `i < j`.
`m` sets the neighborhood radius for conditioning sets; `ugl_penalty` controls
nodewise lasso. Larger `alpha` generally retains more adjacencies. Defaults use
OR-symmetrized lasso and Gaussian Fisher-z tests; the latter require nonsingular
conditional correlations and enough samples (`N > |Z| + 3`). The paper's error
bound requires its assumptions; `alpha` alone is not an unconditional FDR guarantee.
Runtime can grow exponentially. Optional `ugl(X)` and `ci_test(i, j, Z)` callables
supply a graph on `0..p-1` and CI p-values, respectively.

## Reproduce the paper

Run `python -m experiments NAME` from the repository root:

| NAME | Paper output | Fixed configuration |
| --- | --- | --- |
| `faithfulness` | Figure 5(a) | 1,000 SCMs, 10 variables, edge probability .2; 20 sample sizes; α=.1 |
| `examples` | Figure 5(b–d); examples in Figures 2–4 | 10 parameter draws × 20 datasets × 5 sample sizes; mCPC m=1, α=.1, UGL=.01 |
| `pr` | Figure 6(a) | 100 shared SCMs, 10 variables, expected degree 3, N=1,000; mCPC m=1, UGL=.2 |
| `chambers` | Figures 6(b), 7 | 10,000 samples, 11 variables; mCPC m=1, UGL=.2 |
| `scalability` | Table 1 | n=10,12,15,18,20,25; expected degree 1–5; 10 replicates; N=1,000; mCPC m=1, α=.2, UGL=.2 |

`all` runs these suites sequentially. Settings, grids, seeds, and numerical
tolerances are explicit in their modules. PC/CPC use causal-learn's stable
skeleton (`uc_rule=0/2`); GES uses the BIC convention in `mcpc/baselines.py`.
Chambers baseline grids are refined deterministically using evaluated FN/rank gaps.
Its graph panel uses α=.01 for all PC variants and GES penalty λ=.5.

All commands resume; repeat after Ctrl-C to continue. Each completed outcome is
flushed to `results/NAME/runs.jsonl`; timeouts, memory limits, and invalid graphs
are retained, while interrupted jobs are retried. PR figures refresh every
300 seconds (`--plot-every`). `--plot-only` redraws saved partial or complete
results without fitting. Use `--out-dir PATH` for a fresh run or a changed environment.
Saved paper results can be redrawn without the original machine or Chambers CSV.
For a downloaded results bundle: `python -m experiments all --plot-only --out-dir /path/to/bundle/results`.
Exact font rendering and runtimes depend on the platform and library builds.

Results contain seeds, settings, graphs, statuses, timings, cached SEMs, environment
information, PNG/PDF figures, and CSV/Markdown summaries. PR outputs are in
`results/pr/n10/`; examples retain filenames `fig4`, `fig5`, `fig6`; the sole
Chambers graph plot is `cpdag_mcpc.png`. FD/FN and their rates are upper bounds;
precision = 1 − FDR and recall = 1 − FNR. PR means use successfully scored outputs,
with contributing/failure counts reported. AUC trapezoidally integrates the sorted
mean operating points, including the `(0,1)` reference anchor, through the largest
observed recall. Sweep α > 1 is only a tuning parameter.

Four workers (override with `--workers 1`–`4`) each use one BLAS thread, a 3 GiB
cooperative budget, 3.5 GiB stop signal, and 4 GiB hard limit; aggregate limit 12 GiB,
system reserve 4 GiB. Time budgets and measured failures are part of the experiments.

## Causal Chambers data

Download `lt_crl_benchmark_v1.zip` from the [official dataset page](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_crl_benchmark_v1).
Extract `lt_crl_benchmark_v1/buchholz_1_obs/buchholz_1_obs.csv` into
`data/causal_chambers/`, or pass `--data-path /path/to/buchholz_1_obs.csv`.
The loader verifies MD5 `b3ca159596e9047f076a0e711fbf5a13`. Data and results are Git-ignored.

All 10,000 rows of the selected columns `red, green, blue, pol_1, pol_2, current,
angle_1, angle_2, ir_1, ir_2, ir_3` are standardized without log transformation. The 21-adjacency
truth in `experiments/models.py` combines five [actuator-SCM relationships](https://github.com/juangamella/causal-chamber/blob/main/datasets/lt_crl_benchmark_v1/generators/buchholz_1.py)
with 16 selected [camera-configuration physics relationships](https://github.com/juangamella/causal-chamber-package/blob/main/causalchamber/ground_truth/adjacencies/adjacency_lt_camera.csv)
(physics CSV MD5 `f52dfaeb23c39fd90e5ffb5bd8e522c4`). Data: Gamella, Bing & Runge,
*Sanity checking causal representation learning on a simple real-world system*
(ICML 2025), CC BY 4.0. Testbed: Gamella, Peters & Bühlmann,
*Causal chambers as a real-world physical testbed for AI methodology* (2025).
