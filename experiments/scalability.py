"""Level-1 mCPC runtime over graph size and degree."""

from .models import seed

SEED, INSTANCES, N_SAMPLES = 0, 10, 1000
N_NODES = (10, 12, 15, 18, 20, 25)
DEGREES = (1, 2, 3, 4, 5)
WEIGHT_RANGE = (.5, 2.)
MCPC_ALPHA = .2
UGL_COEF_TOL, UGL_MAX_ITER, UGL_TOL = 1e-3, 10000, 1e-3
ALGORITHM_TIMEOUT, METRIC_TIMEOUT = 300, 300


def tasks(data_path=None):
    # Replication-major order gives every cell coverage when interrupted.
    for instance in range(INSTANCES):
        for n in N_NODES:
            for degree in DEGREES:
                spec = dict(kind='random', n_nodes=n, edge_probability=degree/(n-1),
                            weight_range=WEIGHT_RANGE, model_seed=seed(SEED, 7, n, degree, instance),
                            data_seed=seed(SEED, 8, n, degree, instance),
                            n_samples=N_SAMPLES)
                group = dict(experiment='scalability', instance=instance, draw=0,
                             n_nodes=n, degree=degree, n_samples=spec['n_samples'])
                params = dict(m=1, alpha=MCPC_ALPHA, ugl_penalty=.2,
                              ugl_coef_tol=UGL_COEF_TOL, ugl_max_iter=UGL_MAX_ITER,
                              ugl_tol=UGL_TOL, ugl_rule='or')
                yield dict(id=f'scale_{n}_{degree}_{instance}_mcpc_m1_ugl0.2',
                           algorithm='mcpc', params=params, group=group, data_spec=spec,
                           algorithm_timeout=ALGORITHM_TIMEOUT, metric_timeout=METRIC_TIMEOUT)
