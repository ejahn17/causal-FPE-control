"""Parameter sweeps for random-graph precision-recall curves."""

from .models import seed

SEED, INSTANCES, N_NODES, EXPECTED_DEGREE, N_SAMPLES = 0, 100, 10, 3, 1000
WEIGHT_RANGE = (.5, 2.)
PC_ALPHA_GRID = (0, 1e-20, 1e-15, 1e-10, 1e-5, .01, .2, .3, .4, .5, .55,
                 .6, .65, .7, .75, .8, .85, .9, .92, .94, .95, .98, .99,
                 .992, .995, .997, .999, .9999, .99999999)
GES_PENALTY_GRID = (0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1,
                    2, 3, 5, 10, 15, 20, 30, 50, 100, 250, 500, 1000)
MCPC_ALPHA_GRID = (0, 1e-20, 1e-15, 1e-10, 1e-5, .1, 1, 1.5, 2, 3.5, 4,
                   4.2, 4.3, 4.31, 4.32, 4.325, 4.33, 4.34, 4.35, 4.4, 4.5, 5)
MCPC_M = 1
UGL_PENALTY, UGL_COEF_TOL, UGL_MAX_ITER, UGL_TOL = .2, 1e-3, 10000, 1e-3
TIMEOUT = 60


def tasks(data_path=None):
    grids = {'mcpc': MCPC_ALPHA_GRID, 'pc': PC_ALPHA_GRID,
             'cpc': PC_ALPHA_GRID, 'ges': GES_PENALTY_GRID}
    for instance in range(INSTANCES):
        spec = dict(kind='random', n_nodes=N_NODES,
                    edge_probability=EXPECTED_DEGREE / (N_NODES - 1), weight_range=WEIGHT_RANGE,
                    model_seed=seed(SEED, 5, instance), n_samples=N_SAMPLES,
                    data_seed=seed(SEED, 6, instance, 0))
        # Complete parameter rounds across algorithms before the next round/model.
        for index in range(max(map(len, grids.values()))):
            for algorithm, grid in grids.items():
                if index >= len(grid):
                    continue
                value = grid[index]
                params = ({'penalty': value} if algorithm == 'ges'
                          else {'alpha': value, 'stable': True})
                if algorithm == 'mcpc':
                    params = dict(m=MCPC_M, alpha=value, ugl_penalty=UGL_PENALTY,
                                  ugl_coef_tol=UGL_COEF_TOL, ugl_max_iter=UGL_MAX_ITER,
                                  ugl_tol=UGL_TOL, ugl_rule='or')
                group = dict(experiment='pr', instance=instance, draw=0,
                             n_samples=N_SAMPLES, n_nodes=N_NODES, curve=algorithm, value=value)
                yield dict(id=f'pr_{instance}_0_{algorithm}_{value:.12g}',
                           algorithm=algorithm, data_spec=spec, params=params, group=group,
                           algorithm_timeout=TIMEOUT, metric_timeout=TIMEOUT,
                           total_timeout=TIMEOUT)
