"""Light-tunnel graph comparison and adaptive precision-recall sweeps."""

import math

MCPC_ALPHA_GRID = (1e-6, 1e-4, 1e-3, .01, .05, .1, .2, .5, 1,
                   6e-4, 1.22, 1.5, 1.95, 2.5, 3.05, 3.35, 3.65, 4.15, 4.567, 5)
PC_ALPHA_GRID = (0, 1e-20, 1e-15, 1e-10, 1e-6, 1e-5, 1e-4, 1e-3, .01,
                 .05, .1, .2, .3, .4, .5, .55, .6, .65, .7, .75, .8, .85,
                 .9, .92, .94, .95, .98, .99, .992, .995, .997, .999,
                 .9999, .99999999)
GES_PENALTY_GRID = (0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1,
                    2, 3, 5, 10, 15, 20, 30, 50, 100, 250, 500, 1000)
MCPC_M = 1
UGL_PENALTY, UGL_COEF_TOL, UGL_MAX_ITER, UGL_TOL = .2, 1e-3, 10000, 1e-3
MCPC_TIMEOUT, BASELINE_TIMEOUT = 600, 120
REFINE_MAX_RUNS, REFINE_RESOLUTION = 150, .02
REFINE_MAX_FN_STEP, REFINE_MAX_RANK_STEP = 1, 1
REFINE_ZERO_STEP, REFINE_ZERO_FLOOR = 10., 1e-9
DATA_PATH = 'data/causal_chambers/buchholz_1_obs.csv'


def _task(algorithm, value, path):
    params = ({'penalty': value} if algorithm == 'ges'
              else {'alpha': value, 'stable': True})
    if algorithm == 'mcpc':
        params = dict(m=MCPC_M, alpha=value, ugl_penalty=UGL_PENALTY,
                      ugl_coef_tol=UGL_COEF_TOL, ugl_max_iter=UGL_MAX_ITER,
                      ugl_tol=UGL_TOL, ugl_rule='or')
    result = dict(id=f'chambers_{algorithm}_{value:.12g}', algorithm=algorithm,
                  data_spec=dict(kind='chambers', path=str(path), n_samples=None),
                  params=params, group=dict(experiment='chambers', instance=0, draw=0,
                  curve=algorithm, value=value),
                  metric_timeout=MCPC_TIMEOUT if algorithm == 'mcpc' else BASELINE_TIMEOUT,
                  algorithm_timeout=MCPC_TIMEOUT if algorithm == 'mcpc' else BASELINE_TIMEOUT)
    if algorithm == 'mcpc':
        result['total_timeout'] = MCPC_TIMEOUT
    return result


def tasks(data_path=None):
    grids = dict(mcpc=MCPC_ALPHA_GRID, pc=PC_ALPHA_GRID,
                 cpc=PC_ALPHA_GRID, ges=GES_PENALTY_GRID)
    for algorithm, grid in grids.items():
        for value in grid:
            yield _task(algorithm, value, data_path or DATA_PATH)


def _midpoint(algorithm, a, b):
    if a <= 0:
        mid = b / REFINE_ZERO_STEP
        return mid if mid >= REFINE_ZERO_FLOOR else None
    if algorithm in ('pc', 'cpc'):
        ta, tb = math.log(a/(1-a)), math.log(b/(1-b))
        mid = 1 / (1 + math.exp(-(ta+tb)/2))
    else:
        ta, tb = math.log(a), math.log(b)
        mid = math.exp((ta+tb)/2)
    return mid if tb-ta >= REFINE_RESOLUTION else None


def next_points(records):
    """At most one new point per baseline; call again after recording results."""
    proposals = []
    for algorithm in ('pc', 'cpc', 'ges'):
        field = 'penalty' if algorithm == 'ges' else 'alpha'
        base = set(GES_PENALTY_GRID if algorithm == 'ges' else PC_ALPHA_GRID)
        runs = {r['params'][field]: r for r in records
                if r.get('group', {}).get('experiment') == 'chambers'
                and r.get('algorithm') == algorithm}
        if not base <= runs.keys() or len(runs.keys() - base) >= REFINE_MAX_RUNS:
            continue
        points = sorted(v for v, r in runs.items() if r.get('pattern') is not None)
        gaps = []
        for a, b in zip(points, points[1:]):
            left, right = (dict(runs[v].get('metrics') or {}) for v in (a, b))
            for value, metrics in ((a, left), (b, right)):
                if metrics.get('rank') is None:
                    pattern = runs[value]['pattern']
                    metrics['rank'] = len({tuple(sorted(edge)) for field in ('directed', 'undirected')
                                           for edge in pattern.get(field, [])})
            metric = 'fn' if all(r.get('fn') is not None for r in (left, right)) else 'rank'
            if any(r.get(metric) is None for r in (left, right)):
                continue
            threshold = REFINE_MAX_FN_STEP if metric == 'fn' else REFINE_MAX_RANK_STEP
            excess = abs(left[metric] - right[metric]) - threshold
            middle = _midpoint(algorithm, a, b) if excess > 0 else None
            if middle is not None and middle not in runs:
                gaps.append((-excess, a, middle))
        if gaps:
            _, a, middle = min(gaps)
            proposals.append(_task(algorithm, middle, runs[a]['data_spec']['path']))
    return proposals
