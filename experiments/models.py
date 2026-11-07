"""Numerical SEMs and the Causal Chambers observational data."""

import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy import __version__ as scipy_version

SAMPLER_VERSION = 1


def seed(*parts):
    return int(np.random.SeedSequence(parts).generate_state(1)[0])


def covariance(weights, variances):
    transform = np.linalg.inv(np.eye(len(variances)) - weights)
    return (transform * variances) @ transform.T


def _weight(rng, bounds):
    magnitude = rng.uniform(*bounds)
    return magnitude if rng.random() < .5 else -magnitude


def random_model(spec):
    n = spec['n_nodes']
    labels = [f'X{i}' for i in range(n)]
    rng = random.Random(spec['model_seed'])
    order = list(labels)
    rng.shuffle(order)
    edges = [(u, v) for k, u in enumerate(order) for v in order[k + 1:]
             if rng.random() < spec['edge_probability']]
    labels.sort()  # The legacy sampler sorted names, including X10 before X2.
    index = {name: i for i, name in enumerate(labels)}
    rng = random.Random(rng.randrange(2 ** 31))
    variances = np.array([rng.expovariate(1) for _ in labels])
    weights = np.zeros((n, n))
    for u, v in sorted(edges, key=lambda e: 'w_' + ''.join(e)):
        weights[index[v], index[u]] = _weight(rng, spec['weight_range'])
    return weights, variances, labels, [(index[u], index[v]) for u, v in edges]


def example_model(spec):
    """Keep the legacy underdetermined least-squares sampling procedure."""
    name = spec['example']
    if name == 'fig4':
        labels = ['A', 'B', 'X1', 'X2']
        edges = [('A', 'X1'), ('A', 'X2'), ('X1', 'B'),
                 ('X1', 'X2'), ('X2', 'B')]
        residual = lambda w, v: [v[0] * (w[0]*w[2] + w[1]*w[4]
                                          + w[0]*w[3]*w[4])]
    elif name == 'fig5':
        labels = ['A', 'B', 'X1', 'X2']
        edges = [('A', 'B'), ('A', 'X1'), ('A', 'X2'),
                 ('B', 'X1'), ('B', 'X2')]
        residual = lambda w, v: [v[0]*(w[1] + w[0]*w[3]),
                                  v[0]*(w[2] + w[0]*w[4])]
    elif name == 'fig6':
        labels = ['X1', 'X2', 'X3', 'X4']
        edges = [('X1', 'X3'), ('X1', 'X4'), ('X2', 'X3'), ('X3', 'X4')]
        residual = lambda w, v: [v[0]*(w[0]*v[3]
                                          - w[1]*w[3]*(w[2]**2*v[1] + v[2]))]
    else:
        raise ValueError(f'Unknown example: {name}')
    rng = random.Random(spec['model_seed'])
    for _ in range(50):
        variances = np.array([rng.expovariate(1) for _ in labels])
        for _ in range(5):
            start = [_weight(rng, spec['weight_range']) for _ in edges]
            fit = least_squares(lambda w: residual(w, variances), start,
                                xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=500)
            if (np.isfinite(fit.x).all() and variances.min() > 1e-9
                    and np.max(np.abs(residual(fit.x, variances))) <= 1e-9):
                index = {name: i for i, name in enumerate(labels)}
                indexed = [(index[u], index[v]) for u, v in edges]
                weights = np.zeros((len(labels), len(labels)))
                for (u, v), w in zip(indexed, fit.x):
                    weights[v, u] = w
                return weights, variances, labels, indexed
    raise RuntimeError(f'Could not sample constrained model {name}')


CHAMBERS_COLUMNS = ('red', 'green', 'blue', 'pol_1', 'pol_2', 'current',
                    'angle_1', 'angle_2', 'ir_1', 'ir_2', 'ir_3')
CHAMBERS_MD5 = 'b3ca159596e9047f076a0e711fbf5a13'
CHAMBERS_EDGES = (
    ('blue', 'green'), ('blue', 'pol_2'), ('green', 'red'),
    ('green', 'pol_2'), ('pol_1', 'red'),
    *((color, sensor) for color in ('red', 'green', 'blue')
      for sensor in ('current', 'ir_1', 'ir_2', 'ir_3')),
    ('pol_1', 'angle_1'), ('pol_1', 'ir_3'),
    ('pol_2', 'angle_2'), ('pol_2', 'ir_3'))


def _chambers(spec):
    path = Path(spec['path'])
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != CHAMBERS_MD5:
        raise ValueError(f'{path}: expected buchholz_1_obs.csv MD5 {CHAMBERS_MD5}')
    with path.open(newline='') as handle:
        rows = csv.DictReader(handle)
        data = np.array([[float(row[c]) for c in CHAMBERS_COLUMNS] for row in rows])
    if spec.get('n_samples'):
        data = data[:spec['n_samples']]
    if not np.isfinite(data).all() or np.any(data.std(axis=0) == 0):
        raise ValueError('Causal Chambers data contain missing or constant columns')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    index = {name: i for i, name in enumerate(CHAMBERS_COLUMNS)}
    truth = nx.DiGraph()
    truth.add_nodes_from(range(len(CHAMBERS_COLUMNS)))
    truth.add_edges_from((index[u], index[v]) for u, v in CHAMBERS_EDGES)
    return data, truth, {'labels': list(CHAMBERS_COLUMNS), 'md5': digest,
                          'model_key': 'chambers_scm', 'data_path': str(path)}


def materialize(spec, cache_dir):
    """Return data, integer-labelled truth DAG and saved-model metadata."""
    if spec['kind'] == 'chambers':
        return _chambers(spec)
    if spec['kind'] not in ('random', 'example'):
        raise ValueError(f"Unknown data kind: {spec['kind']}")
    model_spec = {k: v for k, v in spec.items() if k not in ('data_seed', 'n_samples')}
    model_spec['sampler'] = dict(version=SAMPLER_VERSION, numpy=np.__version__, scipy=scipy_version,
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(model_spec, sort_keys=True).encode()).hexdigest()[:20]
    path = Path(cache_dir) / (key + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        factory = random_model if spec['kind'] == 'random' else example_model
        weights, variances, labels, edges = factory(spec)
        sigma = covariance(weights, variances)
        scale = 1 / np.sqrt(np.diag(sigma))
        weights = scale[:, None] * weights / scale[None, :]
        variances = variances * scale**2
        sigma = scale[:, None] * sigma * scale[None, :]
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
            temporary = f.name
            np.savez(f, weights=weights, variances=variances, covariance=sigma,
                     labels=np.array(labels), edges=np.array(edges, dtype=int).reshape(-1, 2),
                     specification=json.dumps(model_spec, sort_keys=True))
        os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as model:
        sigma, labels, edges = model['covariance'], model['labels'].tolist(), model['edges']
    truth = nx.DiGraph()
    truth.add_nodes_from(range(len(labels)))
    truth.add_edges_from(map(tuple, edges.tolist()))
    rng = np.random.default_rng(spec['data_seed'])
    data = rng.multivariate_normal(np.zeros(len(labels)), sigma, size=spec['n_samples'])
    return data, truth, {'labels': labels, 'model_key': key, 'model_path': str(path)}
