"""PC-stable, causal-learn's conservative rule (uc_rule=2), and GES."""

from itertools import combinations
from threading import RLock
from unittest.mock import patch

import numpy as np

from .graph import Pattern, triples
from .limits import check_memory
from .orientation import pattern_from_classification, representatives


class InvalidGraph(ValueError):
    """An algorithm output has no represented DAG."""

    def __init__(self, message, metrics=None, pattern=None):
        super().__init__(message)
        self.pattern = pattern
        if metrics is None and pattern is not None:
            skeleton = pattern.skeleton()
            metrics = dict(rank=skeleton.number_of_edges(), n_amb=len(pattern.ambiguous),
                           n_triples=len(triples(skeleton)), n_directed=len(pattern.directed))
        self.metrics = metrics


def _decode(matrix):
    directed, undirected, conflicts = set(), set(), []
    for i, j in combinations(range(len(matrix)), 2):
        endpoints = tuple(matrix[[i, j], [j, i]])
        if endpoints == (-1, 1):
            directed.add((i, j))
        elif endpoints == (1, -1):
            directed.add((j, i))
        elif endpoints == (-1, -1):
            undirected.add((i, j))
        elif endpoints != (0, 0):
            # Preserve adjacency for invalid-output diagnostics and drawings.
            undirected.add((i, j))
            conflicts.append((i, j))
    pattern = Pattern(len(matrix), directed, undirected, set())
    if conflicts:
        raise InvalidGraph(f"Unsupported/conflicting endpoints on edges {conflicts}", pattern=pattern)
    return pattern


def local_score_BIC_from_cov(data, node, parents, parameters=None):
    """NumPy-2-safe Gaussian BIC; lambda=.5 is the textbook penalty."""
    check_memory()
    covariance, n = data
    penalty = .5 if parameters is None else parameters["lambda_value"]
    variance = covariance[node, node]
    if parents:
        cross = covariance[node, parents]
        xx = covariance[np.ix_(parents, parents)]
        try:
            inverse = np.linalg.inv(xx)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(xx)
        variance -= (cross @ inverse @ cross).item()
    variance = max(float(variance), np.finfo(float).eps)
    return -.5 * n * (1 + np.log(variance)) - penalty * (len(parents) + 1) * np.log(n)


_GES_LOCK = RLock()


def fit(data, algorithm: str, params: dict):
    """Return an algorithm Result; CPC representatives are evaluated lazily."""
    from .algorithm import Result

    algorithm = algorithm.lower()
    data = np.asarray(data, dtype=float)
    diagnostics = {"algorithm": algorithm}
    if algorithm in {"pc", "cpc"}:
        from causallearn.search.ConstraintBased.PC import pc

        alpha = float(params.get("alpha", .1))
        if not np.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("PC/CPC alpha must be finite and in [0, 1)")
        # Upstream rejects zero. Its Fisher-z computes 2*(1-CDF), whose
        # positive values exceed this lower limit; p=0 remains dependent.
        effective_alpha = float(np.nextafter(0., 1.)) if alpha == 0 else alpha
        diagnostics.update(alpha_requested=alpha, alpha_effective=effective_alpha)
        cg = pc(data, alpha=effective_alpha, indep_test="fisherz",
                stable=True, uc_rule=2 if algorithm == "cpc" else 0,
                uc_priority=-1, show_progress=False)
        if algorithm == "cpc":
            # Use precisely the classifications used by the original experiments.
            matrix = cg.G.graph
            skeleton = Pattern(data.shape[1], set(),
                               {(i, j) for i, j in combinations(range(data.shape[1]), 2)
                                if matrix[i, j] or matrix[j, i]}, set()).skeleton()
            all_triples = set(triples(skeleton))
            canonical = lambda t: (min(t[0], t[2]), t[1], max(t[0], t[2]))
            colliders = {canonical(t) for t in cg.definite_UC} & all_triples
            noncolliders = ({canonical(t) for t in cg.definite_non_UC} & all_triples) - colliders
            ambiguous = all_triples - colliders - noncolliders
            try:
                pattern = pattern_from_classification(skeleton, colliders, ambiguous)
            except ValueError as error:
                try:
                    pattern = _decode(matrix)
                except InvalidGraph as invalid:
                    pattern = invalid.pattern
                pattern.ambiguous = ambiguous
                raise InvalidGraph(str(error), pattern=pattern) from error
            diagnostics["uc_rule"] = 2
            return Result(pattern, representatives(pattern), diagnostics)
        pattern = _decode(cg.G.graph)
    elif algorithm == "ges":
        import causallearn.search.ScoreBased.GES as ges_module

        # causal-learn has no scorer argument. Restore its global immediately,
        # including on failure; importing this module changes nothing upstream.
        with _GES_LOCK, patch.object(ges_module, "local_score_BIC_from_cov",
                                     local_score_BIC_from_cov):
            output = ges_module.ges(data, score_func="local_score_BIC",
                                    lambda_value=params.get("penalty", .5))
        pattern = _decode(output["G"].graph)
    else:
        raise ValueError(f"Unknown baseline: {algorithm}")
    dag = next(representatives(pattern), None)
    if dag is None:
        raise InvalidGraph(f"{algorithm.upper()} output has no consistent DAG extension", pattern=pattern)
    return Result(pattern, [dag], diagnostics)
