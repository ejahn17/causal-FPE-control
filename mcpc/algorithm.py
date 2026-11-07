"""m-conservative PC (Algorithm 1)."""
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .ci import FisherZ, by_test, neighborhood_graph, test_list
from .graph import Pattern
from .orientation import orient


@dataclass
class Result:
    pattern: Pattern
    representatives: list[nx.DiGraph]
    diagnostics: dict


def mcpc(X, *, m=1, alpha=.1, ugl_penalty=.01, ugl=None, ci_test=None,
         ugl_coef_tol=1e-6, ugl_max_iter=2000, ugl_tol=1e-4, ugl_rule='or'):
    """Return an e-pattern and one DAG per represented equivalence class.

    X has observations in rows. Optional ugl(X) returns an undirected graph
    on 0..p-1; ci_test(i,j,Z) returns a CI p-value. Alpha > 1 is allowed for
    operating-point sweeps, but is not a probability-valued error target.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 4 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a finite observations-by-variables matrix")
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError("m must be a positive integer")
    if not np.isfinite(alpha) or alpha < 0 or ugl_penalty < 0:
        raise ValueError("alpha and ugl_penalty must be nonnegative")
    if ugl_rule != 'or':
        raise ValueError("only OR neighborhood symmetrization is supported")
    if np.any(X.std(axis=0) == 0) and ci_test is None:
        raise ValueError("constant variables are not supported by Fisher-z")
    graph = ugl(X) if ugl is not None else neighborhood_graph(
        X, ugl_penalty, ugl_coef_tol, ugl_max_iter, ugl_tol)
    if graph.is_directed() or set(graph) != set(range(X.shape[1])) or nx.number_of_selfloops(graph):
        raise ValueError("UGL must return a loop-free undirected graph on 0..p-1")
    independent, diagnostics = by_test(lambda: test_list(graph, int(m)),
                                       ci_test if ci_test is not None else FisherZ(X), alpha)
    skeleton = nx.complete_graph(X.shape[1])
    skeleton.remove_edges_from((i, j) for i, j, _ in independent)
    pattern, dags = orient(skeleton, independent)
    diagnostics.update(ugl_edges=sorted(tuple(sorted(e)) for e in graph.edges),
                       skeleton_edges=sorted(skeleton.edges), n_mecs=len(dags))
    return Result(pattern, dags, diagnostics)
