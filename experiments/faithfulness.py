"""Empirical violations of six faithfulness conditions."""

from itertools import combinations
from math import atanh, erfc, sqrt

import networkx as nx
import numpy as np

from .models import seed

SEED, INSTANCES, N_NODES, EDGE_PROBABILITY = 0, 1000, 10, .2
ALPHA, WEIGHT_RANGE = .1, (.5, 2.)
SAMPLE_SIZES = tuple(sorted(set(int(round(x)) for x in np.logspace(2, 4, 20))))
TIMEOUT = 300


def tasks(data_path=None):
    sizes = SAMPLE_SIZES
    for instance in range(INSTANCES):
        spec = dict(kind='random', n_nodes=N_NODES,
                    edge_probability=EDGE_PROBABILITY, weight_range=WEIGHT_RANGE,
                    model_seed=SEED + instance, data_seed=seed(SEED, 9, instance),
                    n_samples=max(sizes))
        yield dict(id=f'faithfulness_{instance}', algorithm='faithfulness', data_spec=spec,
                   params=dict(alpha=ALPHA, sample_sizes=sizes),
                   group=dict(experiment='faithfulness', instance=instance, draw=0,
                              n_nodes=spec['n_nodes'], n_samples=max(sizes)),
                   algorithm_timeout=TIMEOUT, metric_timeout=TIMEOUT)


def _subsets(vertices):
    for k in range(len(vertices) + 1):
        yield from combinations(vertices, k)


def _pvalue(sigma, i, j, z, n):
    if n - len(z) - 3 <= 0:
        return 1.
    indices = [i, j, *z]
    try:
        precision = np.linalg.inv(sigma[np.ix_(indices, indices)])
    except np.linalg.LinAlgError:
        return 1.
    denominator = sqrt(max(precision[0, 0]*precision[1, 1], 0))
    if denominator == 0:
        return 1.
    rho = float(np.clip(-precision[0, 1]/denominator, -1+1e-15, 1-1e-15))
    return erfc(abs(atanh(rho))*sqrt(n-len(z)-3)/sqrt(2))


def evaluate(data, truth, params):
    nodes = sorted(truth.nodes)
    skeleton = truth.to_undirected()
    tests = dict(faithfulness=[], adjacency_faithfulness=[], triangle_faithfulness=[])
    for i, j in combinations(nodes, 2):
        others = [v for v in nodes if v not in (i, j)]
        common = set(skeleton[i]) & set(skeleton[j]) if skeleton.has_edge(i, j) else set()
        for z in _subsets(others):
            item = (i, j, z)
            if not nx.is_d_separator(truth, {i}, {j}, set(z)):
                tests['faithfulness'].append(item)
            if skeleton.has_edge(i, j):
                tests['adjacency_faithfulness'].append(item)
            if any((truth.has_edge(i, k) and truth.has_edge(j, k)) == (k in z)
                   for k in common):
                tests['triangle_faithfulness'].append(item)
    rows = []
    for n in params['sample_sizes']:
        sigma = np.cov(data[:n], rowvar=False, ddof=1)
        cache = {}
        def accepted(item):
            if item not in cache:
                cache[item] = _pvalue(sigma, *item, n) > params['alpha']
            return cache[item]
        row = {'n_samples': n}
        row.update({condition: int(any(accepted(t) for t in required))
                    for condition, required in tests.items()})
        bad = nx.DiGraph()
        for i, j in truth.edges:
            pair = (min(i, j), max(i, j), tuple(v for v in nodes if v not in (i, j)))
            if accepted(pair):
                bad.add_edge(i, j)
        largest = max((bad.subgraph(nx.descendants(bad, v) | {v}).number_of_edges()
                       for v in bad if bad.in_degree(v) == 0), default=0)
        row.update({f'm{m}': int(largest >= m) for m in (1, 2, 3)})
        rows.append(row)
    return dict(rows=rows, model_count=1)
