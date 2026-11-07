"""Paper Algorithms 5–6, with deletion-stable, chordless activation witnesses."""

import networkx as nx

from .assignments import Assignments
from .graph import Pattern, triple, triples
from .limits import check_memory


def _edge(i, j):
    return (min(i, j), max(i, j))


def _close(skeleton, colliders, allowed, fixed=()):
    """Parallel Meek closure; each arrow carries its forcing triples."""
    arrows = {edge: set() for edge in fixed}
    pending = {}
    for t in sorted(colliders):
        i, k, j = t
        for edge in ((i, k), (j, k)):
            pending.setdefault(edge, set()).add(t)

    def undirected(i, j):
        return skeleton.has_edge(i, j) and (i, j) not in arrows and (j, i) not in arrows

    while True:
        for (i, j), provenance in sorted(pending.items()):
            if (j, i) in arrows or (j, i) in pending:
                return None, provenance | arrows.get((j, i), set()) | pending.get((j, i), set())
        for edge, provenance in pending.items():
            arrows.setdefault(edge, set()).update(provenance)
        for t in sorted(allowed - colliders):
            i, k, j = t
            if (i, k) in arrows and (j, k) in arrows:
                return None, {t} | arrows[i, k] | arrows[j, k]
        graph = nx.DiGraph()
        graph.add_nodes_from(skeleton)
        graph.add_edges_from(sorted(arrows))
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            cycle = None
        if cycle:
            return None, set().union(*(arrows[edge] for edge in cycle))

        pending = {}

        def propose(edge, supports, used=()):
            used = set(used)
            if used <= allowed:
                provenance = used.union(*(arrows[source] for source in supports))
                pending.setdefault(edge, set()).update(provenance)

        for i, k, j in sorted(allowed):
            t = (i, k, j)
            for a, b in ((i, j), (j, i)):
                if (a, k) in arrows and undirected(k, b):       # R1
                    propose((k, b), [(a, k)], [t])
            if not (undirected(k, i) and undirected(k, j)):
                continue
            for c in sorted(skeleton[k]):
                if not undirected(k, c) or c in (i, j):
                    continue
                other = triple(i, c, j)
                if other not in allowed:
                    continue
                if (i, c) in arrows and (j, c) in arrows:       # R3
                    propose((k, c), [(i, c), (j, c)], [t, other])
                for a, b in ((i, j), (j, i)):
                    if (a, c) in arrows and (c, b) in arrows:   # R4
                        propose((k, b), [(a, c), (c, b)], [t, other])
        for i, j in sorted(skeleton.edges()):
            if not undirected(i, j):
                continue
            for a, b in ((i, j), (j, i)):
                for k in sorted(skeleton[a]):
                    if (a, k) in arrows and (k, b) in arrows:   # R2
                        propose((a, b), [(a, k), (k, b)])
        if not pending:
            return arrows, set()


def _extend(skeleton, arrows):
    """Dor–Tarsi extension: remove eligible sinks without adding colliders."""
    directed = set(arrows)
    children = {v: {j for i, j in directed if i == v} for v in skeleton}
    parents = {v: {i for i, j in directed if j == v} for v in skeleton}
    undirected = {v: set(skeleton[v]) - children[v] - parents[v] for v in skeleton}
    remaining = set(skeleton)
    while remaining:
        removed = False
        for v in sorted(remaining):
            neighbors = undirected[v] & remaining
            adjacent = (parents[v] & remaining) | neighbors
            if children[v] & remaining or any(adjacent - {w} - set(skeleton[w]) for w in neighbors):
                continue
            directed.update((w, v) for w in neighbors)
            remaining.remove(v)
            removed = True
        if not removed:
            return None
    dag = nx.DiGraph()
    dag.add_nodes_from(skeleton)
    dag.add_edges_from(sorted(directed))
    return dag


def _realize(skeleton, assignment, fixed=()):
    allowed = set(assignment)
    colliders = {t for t, value in assignment.items() if value}
    arrows, conflict = _close(skeleton, colliders, allowed, fixed)
    if arrows is None:
        return None, conflict
    undirected = nx.Graph()
    undirected.add_nodes_from(skeleton)
    undirected.add_edges_from((i, j) for i, j in skeleton.edges()
                             if (i, j) not in arrows and (j, i) not in arrows)
    for component in nx.connected_components(undirected):
        if len(component) < 4:
            continue
        for cycle in nx.chordless_cycles(skeleton.subgraph(component)):
            if len(cycle) >= 4:
                return None, {triple(cycle[i - 1], v, cycle[(i + 1) % len(cycle)])
                              for i, v in enumerate(cycle)}
    dag = _extend(skeleton, arrows)
    if dag is None:
        # Extra fixed arrows can make an MPDAG non-extendable; the whole
        # assignment is a sound certificate relative to those fixed arrows.
        if fixed:
            return None, set(assignment)
        raise RuntimeError("Meek-closed assignment has no consistent extension")
    return dag, set()


def realize(skeleton, assignment):
    """Return a realizing DAG, or a conflicting subset of assigned triples."""
    if set(assignment) != set(triples(skeleton)):
        raise ValueError("assign every unshielded triple exactly once")
    return _realize(skeleton, assignment)


def pattern_from_classification(skeleton, colliders, ambiguous):
    """Orient definite colliders and close Meek rules avoiding ambiguous triples."""
    allowed = set(triples(skeleton)) - set(ambiguous)
    arrows, conflict = _close(skeleton, set(colliders) - set(ambiguous), allowed)
    if arrows is None:
        raise ValueError(f"inconsistent collider assignment: {sorted(conflict)}")
    directed = set(arrows)
    undirected = {_edge(i, j) for i, j in skeleton.edges()
                  if (i, j) not in directed and (j, i) not in directed}
    return Pattern(len(skeleton), directed, undirected, set(ambiguous))


def representatives(pattern):
    """Yield one DAG per represented equivalence class, respecting fixed arrows."""
    skeleton = pattern.skeleton()
    variables = triples(skeleton)
    forced = {t: (t[0], t[1]) in pattern.directed and (t[2], t[1]) in pattern.directed
              for t in variables if t not in pattern.ambiguous}
    enumeration = Assignments(variables)
    enumeration.force(forced)
    while (assignment := enumeration.next()) is not None:
        dag, conflict = _realize(skeleton, assignment, pattern.directed)
        if dag is None:
            enumeration.forbid({t: assignment[t] for t in conflict})
        else:
            yield dag


def _activation_paths(dag, collider, conditioned):
    """Induced directed paths, stopping at the first conditioned vertex."""
    ancestors = set(conditioned).union(*(nx.ancestors(dag, z) for z in conditioned))
    def visit(path):
        if path[-1] in conditioned:
            yield tuple(path)
            return
        for child in sorted(set(dag.successors(path[-1])) & ancestors):
            if all(not (dag.has_edge(child, v) or dag.has_edge(v, child)) for v in path[:-1]):
                yield from visit(path + [child])
    yield from visit([collider])


def _witness_triples(dag, path, conditioned, candidates):
    """Two required edges must coexist in one witness, not across alternatives."""
    base = {_edge(i, j) for i, j in zip(path, path[1:])}
    missing = {t: tuple(edge for edge in (_edge(t[0], t[1]), _edge(t[1], t[2]))
                        if edge not in base) for t in candidates}
    marked = {t for t, edges in missing.items() if not edges}
    remaining = {t: edges for t, edges in missing.items() if edges}
    supplied = {t: [0] * len(edges) for t, edges in remaining.items()}
    colliders = [v for i, v, j in zip(path, path[1:], path[2:])
                 if dag.has_edge(i, v) and dag.has_edge(j, v)]
    for index, collider in enumerate(colliders):
        for activation in _activation_paths(dag, collider, conditioned):
            check_memory()
            edges = {_edge(i, j) for i, j in zip(activation, activation[1:])}
            for t, required in list(remaining.items()):
                for k, edge in enumerate(required):
                    if edge in edges:
                        supplied[t][k] |= 1 << index
                coexist = all(edge in edges for edge in required)
                if len(required) == 2:
                    a, b = supplied[t]
                    coexist |= bool(a and b and (a | b).bit_count() > 1)
                if coexist:
                    marked.add(t)
                    del remaining[t]
            if not remaining:
                return marked
    return marked


def _witness_ambiguities(dag, i, j, conditioned, candidates):
    conditioned = set(conditioned)
    ancestors = conditioned | set().union(*(nx.ancestors(dag, z) for z in conditioned))
    marked = set()
    for path in nx.all_simple_paths(dag.to_undirected(), i, j):
        check_memory()
        active = all((v in ancestors if dag.has_edge(a, v) and dag.has_edge(b, v)
                      else v not in conditioned)
                     for a, v, b in zip(path, path[1:], path[2:]))
        if active:
            marked |= _witness_triples(dag, path, conditioned, candidates - marked)
            if marked == candidates:
                break
    return marked


def orient(skeleton, independences):
    """OrientEPattern: exhaust conflict and CI-witness ambiguity escalations."""
    variables = triples(skeleton)
    independences = sorted({(min(i, j), max(i, j), tuple(sorted(z)))
                            for i, j, z in independences})
    separating = {}
    for i, j, z in independences:
        separating.setdefault((i, j), []).append(z)
    forced = {}
    for t in variables:
        i, k, j = t
        membership = {k in z for z in separating.get((i, j), [])}
        if len(membership) < 2:
            forced[t] = membership == {False}
    enumeration = Assignments(variables)
    enumeration.force(forced)
    dags, pending = [], []
    while True:
        check_memory()
        assignment = enumeration.next()
        if assignment is not None:
            dag, conflict = _realize(skeleton, assignment)
            if dag is not None:
                dags.append(dag)
                if independences:
                    pending.append([dag, len(independences) - 1])
            else:
                enumeration.forbid({t: assignment[t] for t in conflict})
                forced = {t: value for t, value in forced.items() if t not in conflict}
                enumeration.force(forced)
            continue
        if not pending:
            break
        dag, index = pending[-1]
        pending[-1][1] -= 1
        if index == 0:
            pending.pop()
        if not forced:
            continue
        i, j, z = independences[index]
        if nx.is_d_separator(dag, {i}, {j}, set(z)):
            continue
        new = _witness_ambiguities(dag, i, j, z, set(forced))
        if new:
            forced = {t: value for t, value in forced.items() if t not in new}
            enumeration.force(forced)
    if not dags:
        raise RuntimeError("orientation produced no represented DAG")
    pattern = pattern_from_classification(skeleton, {t for t, value in forced.items() if value},
                                          set(variables) - set(forced))
    return pattern, dags
