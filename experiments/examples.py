"""The three four-variable motivating examples (legacy fig4/fig5/fig6)."""

from .models import seed

SEED = 0
PARAMETER_DRAWS = 10
DATA_DRAWS = 20
SAMPLE_SIZES = (100, 300, 1000, 3000, 10000)
WEIGHT_RANGE = (.5, 2.)
ALPHA = .1
MCPC_M = 1
UGL_PENALTY, UGL_COEF_TOL, UGL_MAX_ITER, UGL_TOL = .01, 1e-6, 2000, 1e-4
TIMEOUT = 120


def tasks(data_path=None):
    for e, example in enumerate(('fig4', 'fig5', 'fig6')):
        for instance in range(PARAMETER_DRAWS):
            for draw in range(DATA_DRAWS):
                for n_samples in SAMPLE_SIZES:
                    group = dict(experiment='examples', example=example, instance=instance,
                                 draw=draw, n_samples=n_samples)
                    spec = dict(kind='example', example=example,
                                model_seed=seed(SEED, 1, e, instance),
                                data_seed=seed(SEED, 2, e, instance, draw, n_samples),
                                n_samples=n_samples, weight_range=WEIGHT_RANGE)
                    for algorithm in ('mcpc', 'pc', 'cpc', 'ges'):
                        params = ({'penalty': .5} if algorithm == 'ges'
                                  else {'alpha': ALPHA, 'stable': True})
                        if algorithm == 'mcpc':
                            params = dict(m=MCPC_M, alpha=ALPHA, ugl_penalty=UGL_PENALTY,
                                          ugl_coef_tol=UGL_COEF_TOL, ugl_max_iter=UGL_MAX_ITER,
                                          ugl_tol=UGL_TOL, ugl_rule='or')
                        yield dict(id=f'{example}_{instance}_{draw}_{n_samples}_{algorithm}',
                                   algorithm=algorithm, data_spec=spec, params=params,
                                   group=group, algorithm_timeout=TIMEOUT, metric_timeout=TIMEOUT)
