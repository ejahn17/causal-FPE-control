"""Poset error upper bounds and CPDAG SHD (minimized over represented MECs)."""

from itertools import combinations, product

import networkx as nx

from .graph import Pattern, cpdag, triples
from .limits import check_memory


def _adjacencies(dag):
    return {tuple(sorted(edge)) for edge in dag.edges()}


def _collider(dag, triple):
    i, k, j = triple
    return dag.has_edge(i, k) and dag.has_edge(j, k)


def _aligned_member(dag, order, pattern=None):
    """Align reversible components to the truth order when no collider is added."""
    pattern = cpdag(dag) if pattern is None else pattern
    graph = nx.Graph(list(pattern.undirected))
    aligned = dag.copy()
    for nodes in nx.connected_components(graph):
        edges = list(graph.subgraph(nodes).edges())
        candidate = nx.DiGraph((u, v) if order[u] < order[v] else (v, u)
                               for u, v in edges)
        if any(not graph.has_edge(a, b) for k in candidate
               for a, b in combinations(candidate.predecessors(k), 2)):
            continue
        aligned.remove_edges_from((u, v) for u, v in dag.edges()
                                  if u in nodes and v in nodes)
        aligned.add_edges_from(candidate.edges())
    return aligned


def _smart_member(dag, truth, deleted_edge):
    """Before deletion, favor orientations matching newly unshielded triples."""
    pattern = cpdag(dag)
    skeleton, target = pattern.skeleton(), truth.to_undirected()
    u, v = deleted_edge
    centers = (set(skeleton[u]) & set(skeleton[v])
               & set(target[u]) & set(target[v]))
    if not centers:
        return dag
    reversible = nx.Graph(list(pattern.undirected))
    result = dag.copy()
    for nodes in nx.connected_components(reversible):
        relevant = [(u, k, v) for k in sorted(centers & nodes)]
        edges = sorted(tuple(sorted(e)) for e in reversible.subgraph(nodes).edges())
        # This cap affects only the bound's tightness, never the discovery algorithm.
        if not relevant or len(edges) > 14:
            continue
        best, best_score = result, sum(_collider(result, t) == _collider(truth, t)
                                      for t in relevant)
        if best_score == len(relevant):
            continue
        fixed = result.copy()
        fixed.remove_edges_from((a, b) for a, b in result.edges()
                                if a in nodes and b in nodes)
        for bits in product((False, True), repeat=len(edges)):
            check_memory()
            candidate = fixed.copy()
            candidate.add_edges_from((a, b) if forward else (b, a)
                                     for (a, b), forward in zip(edges, bits))
            if not nx.is_directed_acyclic_graph(candidate):
                continue
            if any(not skeleton.has_edge(a, b)
                   and not ((a, k) in pattern.directed and (b, k) in pattern.directed)
                   for k in nodes for a, b in combinations(candidate.predecessors(k), 2)):
                continue
            score = sum(_collider(candidate, t) == _collider(truth, t) for t in relevant)
            if score > best_score:
                best, best_score = candidate, score
                if score == len(relevant):
                    break
        result = best
    return result


def _parent_statements(dag):
    order = list(nx.lexicographical_topological_sort(dag))
    return [(i, j, frozenset(dag.predecessors(j)))
            for position, j in enumerate(order) for i in order[:position]
            if not dag.has_edge(i, j)]


def _cut(dag, i, j, conditioned):
    ancestors = {i, j} | set(conditioned)
    for node in tuple(ancestors):
        ancestors.update(nx.ancestors(dag, node))
    moral = nx.moral_graph(dag.subgraph(ancestors))
    moral.remove_nodes_from(conditioned)
    removed = set()
    for u, v in sorted(tuple(sorted(e)) for e in nx.minimum_edge_cut(moral, i, j)):
        if dag.has_edge(u, v):
            removed.add((u, v))
        elif dag.has_edge(v, u):
            removed.add((v, u))
        else:
            children = set(dag[u]) & set(dag[v]) & ancestors
            removed.add((u, min(children)))
    return removed


def similarity(estimate, truth, pattern=None):
    """Return a lower bound on common poset rank; FD/FN share this valuation."""
    order = {node: i for i, node in enumerate(nx.lexicographical_topological_sort(truth))}
    current = _aligned_member(estimate, order, pattern)
    target = truth
    if current.number_of_edges() > truth.number_of_edges():
        current, target = truth.copy(), current
    for edge in sorted(_adjacencies(current) - _adjacencies(target)):
        current = _smart_member(current, target, edge)
        u, v = edge
        current.remove_edge(u, v) if current.has_edge(u, v) else current.remove_edge(v, u)
    for i, j, conditioned in _parent_statements(target):
        while not nx.is_d_separator(current, {i}, {j}, set(conditioned)):
            check_memory()
            removed = _cut(current, i, j, conditioned)
            if not removed:
                raise RuntimeError("No cut for a d-connected pair")
            current.remove_edges_from(removed)
    return current.number_of_edges()


def shd(left: Pattern, right: Pattern):
    """One edit for any absent/directed/undirected state mismatch per pair."""
    def state(pattern, i, j):
        if (i, j) in pattern.undirected:
            return 3
        return 1 if (i, j) in pattern.directed else 2 if (j, i) in pattern.directed else 0
    if left.n != right.n:
        raise ValueError("Graphs must have the same vertices")
    return sum(state(left, i, j) != state(right, i, j)
               for i, j in combinations(range(left.n), 2))


def evaluate(pattern: Pattern, representatives, truth: nx.DiGraph) -> dict:
    """FD/FN/FDR/FNR are upper bounds; precision/recall are lower bounds.

    SHD is exact over the supplied complete represented-MEC family. An empty
    family raises InvalidGraph with its structural metrics attached.
    """
    true_pattern = cpdag(truth)
    estimated = {tuple(sorted(e)) for e in pattern.skeleton().edges()}
    actual = _adjacencies(truth)
    rank, true_rank, shared = len(estimated), len(actual), len(estimated & actual)
    result = dict(rank=rank, n_amb=len(pattern.ambiguous),
                  n_triples=len(triples(pattern.skeleton())), n_directed=len(pattern.directed),
                  adj_precision=shared / rank if rank else 1.0,
                  adj_recall=shared / true_rank if true_rank else 1.0,
                  dir_correct=len(pattern.directed & true_pattern.directed),
                  dir_wrong=sum(tuple(sorted(e)) in actual and e not in true_pattern.directed
                                for e in pattern.directed),
                  fd=None, fn=None, fdr=None, fnr=None, precision=None, recall=None, shd=None)
    best_rho, best_shd, found = 0, float("inf"), False
    for dag in representatives:
        check_memory()
        found = True
        essential = cpdag(dag)
        best_shd = min(best_shd, shd(essential, true_pattern))
        if best_rho < shared:
            best_rho = max(best_rho, similarity(dag, truth, essential))
        if best_rho == shared and best_shd == 0:
            break
    if not found:
        from .baselines import InvalidGraph
        raise InvalidGraph("Output represents no DAG", metrics=result, pattern=pattern)
    fd, fn = rank - best_rho, true_rank - best_rho
    fdr, fnr = fd / max(rank, 1), fn / max(true_rank, 1)
    result.update(fd=fd, fn=fn, fdr=fdr, fnr=fnr, precision=1 - fdr,
                  recall=1 - fnr, shd=int(best_shd))
    return result
