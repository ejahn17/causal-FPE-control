"""Gaussian CI tests, neighborhood selection, and Algorithm 4."""
from array import array
from itertools import combinations
from math import fsum

import networkx as nx
import numpy as np
from scipy.special import ndtr
from sklearn.linear_model import Lasso

from .limits import check_memory


class FisherZ:
    def __init__(self, data):
        self.n = len(data)
        self.correlation = np.corrcoef(data, rowvar=False)

    def __call__(self, i, j, conditioned=()):
        indices = (i, j, *conditioned)
        if self.n <= len(conditioned) + 3:
            raise ValueError("Fisher-z requires N > |conditioning set| + 3")
        inverse = np.linalg.inv(self.correlation[np.ix_(indices, indices)])
        r = -inverse[0, 1] / np.sqrt(inverse[0, 0] * inverse[1, 1])
        r = np.clip(r, -1 + 1e-15, 1 - 1e-15)
        z = np.sqrt(self.n - len(conditioned) - 3) * abs(np.arctanh(r))
        # Match causal-learn, including its floating-point zero p-values.
        return float(2 * (1 - ndtr(z)))


def neighborhood_graph(data, penalty=.01, coef_tol=1e-6,
                       max_iter=2000, tol=1e-4):
    centered = data - data.mean(axis=0)
    graph = nx.empty_graph(data.shape[1])
    for j in graph:
        columns = [i for i in graph if i != j]
        if not columns:
            continue
        model = Lasso(alpha=penalty, fit_intercept=False,
                      max_iter=max_iter, tol=tol).fit(centered[:, columns], centered[:, j])
        graph.add_edges_from((i, j) for i, weight in zip(columns, model.coef_)
                             if abs(weight) > coef_tol)
    return graph


def test_list(graph, m):
    neighborhoods = {i: set(nx.single_source_shortest_path_length(graph, i, cutoff=m))
                     for i in graph}
    for i, j in combinations(sorted(graph), 2):
        left = neighborhoods[i] - {i, j}
        right = neighborhoods[j] - {i, j}
        for size in range(len(left) + 1):
            for conditioned in combinations(sorted(left), size):
                yield i, j, conditioned
        for size in range(1, len(right) + 1):
            for conditioned in combinations(sorted(right), size):
                if not left.issuperset(conditioned):
                    yield i, j, conditioned


def by_test(factory, pvalue, alpha):
    values, maxima = array('d'), {}
    for i, j, conditioned in factory():
        check_memory()
        value = float(pvalue(i, j, conditioned))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("CI test returned an invalid p-value")
        values.append(value)
        maxima[i, j] = max(maxima.get((i, j), 0), value)
    count = len(maxima)
    denominator = count * fsum(1 / k for k in range(1, count + 1))
    k = max((k for k, value in enumerate(sorted(maxima.values()), 1)
             if value <= k * alpha / denominator), default=0)
    threshold = k * alpha / denominator if count else 0.0
    independent = []
    position = 0
    for position, triple in enumerate(factory(), 1):
        check_memory()
        if values[position - 1] > threshold:
            independent.append(triple)
    if position != len(values):
        raise ValueError("test factory must be deterministic")
    return independent, dict(n_tests=len(values), n_independences=len(independent),
                             threshold=threshold, selected_pairs=k)
