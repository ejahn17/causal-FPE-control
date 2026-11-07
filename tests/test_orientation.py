"""Small exhaustive oracles independent of the production orientation code."""

import itertools as it
import json
import random
import unittest

import networkx as nx
import numpy as np

from mcpc.assignments import Assignments
from mcpc.graph import Pattern, cpdag, triple, triples
from mcpc.orientation import (orient, realize, representatives, _activation_paths,
                              _witness_triples, _witness_ambiguities)


def all_dags(skeleton):
    edges = sorted(skeleton.edges())
    for flips in it.product((False, True), repeat=len(edges)):
        dag = nx.DiGraph()
        dag.add_nodes_from(skeleton)
        dag.add_edges_from((j, i) if flip else (i, j) for (i, j), flip in zip(edges, flips))
        if nx.is_directed_acyclic_graph(dag):
            yield dag


def assignment(dag, variables):
    return {t: dag.has_edge(t[0], t[1]) and dag.has_edge(t[2], t[1]) for t in variables}


def signature(dag, variables):
    return tuple(assignment(dag, variables).values())


def skeleton(n, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph


class TestAssignments(unittest.TestCase):
    def test_incremental_against_truth_table(self):
        rng = random.Random(184)
        for trial in range(30):
            keys = list(range(6))
            solver = Assignments(keys)
            forced = {key: bool(rng.getrandbits(1)) for key in keys[:3]}
            solver.force(forced)
            forbidden, tried = [], set()
            for step in range(90):
                expected = next((values for values in it.product((False, True), repeat=len(keys))
                                 if values not in tried
                                 and all(values[k] == value for k, value in forced.items())
                                 and not any(all(values[k] == value for k, value in f.items())
                                             for f in forbidden)), None)
                actual = solver.next()
                self.assertEqual(None if actual is None else tuple(actual.values()), expected,
                                 (trial, step, forced, forbidden))
                if actual is not None:
                    tried.add(tuple(actual.values()))
                    if rng.random() < .3:
                        subset = rng.sample(keys, rng.randrange(1, 4))
                        conflict = {key: actual[key] for key in subset}
                        forbidden.append(conflict)
                        solver.forbid(conflict)
                if forced and (actual is None or rng.random() < .15):
                    del forced[next(iter(forced))]
                    solver.force(forced)
                elif actual is None:
                    break

    def test_empty_assignment(self):
        solver = Assignments([])
        self.assertEqual(solver.next(), {})
        self.assertIsNone(solver.next())


class TestRealization(unittest.TestCase):
    def test_numpy_indices_serialize(self):
        pattern = Pattern(np.int64(3), {(np.int64(0), np.int64(1))}, {(1, 2)})
        self.assertEqual(Pattern.from_dict(json.loads(json.dumps(pattern.to_dict()))), pattern)

    def test_all_four_node_skeletons_and_assignments(self):
        edges = list(it.combinations(range(4), 2))
        for mask in range(1 << len(edges)):
            graph = skeleton(4, [edge for i, edge in enumerate(edges) if mask >> i & 1])
            variables = triples(graph)
            feasible = {signature(dag, variables) for dag in all_dags(graph)}
            for values in it.product((False, True), repeat=len(variables)):
                requested = dict(zip(variables, values))
                dag, conflict = realize(graph, requested)
                self.assertEqual(dag is not None, values in feasible, (list(graph.edges()), requested))
                if dag is not None:
                    self.assertEqual(signature(dag, variables), values)
                    self.assertEqual(set(dag.to_undirected().edges()), set(graph.edges()))
                else:
                    self.assertTrue(conflict)
                    indices = [variables.index(t) for t in conflict]
                    self.assertFalse(any(all(other[i] == values[i] for i in indices)
                                         for other in feasible), (requested, conflict))

    def test_cpdag_against_all_equivalent_dags(self):
        graph = skeleton(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
        dags = list(all_dags(graph))
        variables = triples(graph)
        for dag in dags:
            equivalent = [other for other in dags if signature(other, variables) == signature(dag, variables)]
            common = set.intersection(*(set(other.edges()) for other in equivalent))
            pattern = cpdag(dag)
            self.assertEqual(pattern.directed, common)
            self.assertEqual(pattern, Pattern.from_dict(pattern.to_dict()))

    def test_five_node_conflict_certificates(self):
        rng = random.Random(122)
        for _ in range(12):
            graph = skeleton(5, [edge for edge in it.combinations(range(5), 2) if rng.random() < .6])
            variables = triples(graph)
            feasible = {signature(dag, variables) for dag in all_dags(graph)}
            for values in it.product((False, True), repeat=len(variables)):
                dag, conflict = realize(graph, dict(zip(variables, values)))
                self.assertEqual(dag is not None, values in feasible)
                if dag is None:
                    indices = [variables.index(t) for t in conflict]
                    self.assertFalse(any(all(other[i] == values[i] for i in indices) for other in feasible))

    def test_fixed_direction_within_equivalence_class(self):
        pattern = Pattern(3, {(0, 1)}, {(1, 2)})
        dags = list(representatives(pattern))
        self.assertEqual(len(dags), 1)
        self.assertEqual(set(dags[0].edges()), {(0, 1), (1, 2)})


class TestWitnesses(unittest.TestCase):
    def test_alternative_branches_do_not_form_one_witness(self):
        dag = nx.DiGraph([(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)])
        marked = _witness_ambiguities(dag, 0, 1, {5}, set(triples(dag.to_undirected())))
        self.assertNotIn((3, 2, 4), marked)
        self.assertNotIn((3, 5, 4), marked)
        self.assertIn((0, 2, 3), marked)
        self.assertIn((0, 2, 4), marked)

    def test_longer_activation_survives_vertex_deletion(self):
        dag = nx.DiGraph([(0, 2), (1, 2), (2, 3), (3, 6), (2, 4), (4, 5), (5, 6)])
        paths = set(_activation_paths(dag, 2, {6}))
        self.assertEqual(paths, {(2, 3, 6), (2, 4, 5, 6)})
        residual = dag.copy()
        residual.remove_node(3)
        self.assertFalse(nx.is_d_separator(residual, {0}, {1}, {6}))
        marked = _witness_ambiguities(dag, 0, 1, {6}, set(triples(dag.to_undirected())))
        self.assertIn((0, 2, 4), marked)
        self.assertIn((2, 4, 5), marked)
        self.assertEqual(list(_activation_paths(dag, 2, {2, 6})), [(2,)])

    def test_activation_paths_are_induced_and_first_hit(self):
        dag = nx.DiGraph([(0, 1), (1, 2), (0, 2), (2, 3)])
        self.assertEqual(set(_activation_paths(dag, 0, {2, 3})), {(0, 2)})

    def test_cooccurrence_matches_explicit_product(self):
        rng = random.Random(9)
        for _ in range(60):
            dag = nx.DiGraph()
            dag.add_nodes_from(range(6))
            dag.add_edges_from(edge for edge in it.combinations(range(6), 2) if rng.random() < .45)
            candidates = set(triples(dag.to_undirected()))
            i, j = rng.sample(range(6), 2)
            z = set(rng.sample(sorted(set(range(6)) - {i, j}), 2))
            ancestors = z | set().union(*(nx.ancestors(dag, v) for v in z))
            for path in nx.all_simple_paths(dag.to_undirected(), i, j):
                centers = [v for a, v, b in zip(path, path[1:], path[2:])
                           if dag.has_edge(a, v) and dag.has_edge(b, v)]
                if any(v not in ancestors for v in centers) or any(v in z for v in path[1:-1] if v not in centers):
                    continue
                choices = [list(_activation_paths(dag, v, z)) for v in centers]
                expected = set()
                for activation in it.product(*choices):
                    edges = {tuple(sorted(edge)) for p in [path, *activation] for edge in zip(p, p[1:])}
                    expected |= {t for t in candidates if tuple(sorted(t[:2])) in edges
                                 and tuple(sorted(t[1:])) in edges}
                self.assertEqual(_witness_triples(dag, path, z, candidates), expected)


class TestOrientation(unittest.TestCase):
    def test_paper_four_node_examples(self):
        cases = [
            ([(0, 1, 1), (0, 2, 2), (1, 2, 1), (1, 3, 1), (2, 3, -1/3)], 5, 2),
            ([(0, 1, 1), (0, 2, -1), (1, 2, 1), (0, 3, -2), (1, 3, 2)], 3, 3),
            ([(0, 2, 2), (0, 3, 1), (2, 3, 1), (1, 2, 1)], 3, 2),
        ]
        for edges, expected_edges, expected_ambiguities in cases:
            weights = np.zeros((4, 4))
            for i, j, value in edges:
                weights[j, i] = value
            transform = np.linalg.inv(np.eye(4) - weights)
            covariance = transform @ transform.T
            independent = []
            for i, j in it.combinations(range(4), 2):
                rest = sorted(set(range(4)) - {i, j})
                for size in range(3):
                    for z in it.combinations(rest, size):
                        value = covariance[i, j]
                        if z:
                            value -= covariance[i, list(z)] @ np.linalg.solve(
                                covariance[np.ix_(z, z)], covariance[list(z), j])
                        if abs(value) < 1e-10:
                            independent.append((i, j, z))
            graph = nx.complete_graph(4)
            graph.remove_edges_from((i, j) for i, j, _ in independent)
            pattern, dags = orient(graph, independent)
            self.assertEqual(len(pattern.undirected), expected_edges)
            self.assertEqual(len(pattern.ambiguous), expected_ambiguities)
            self.assertFalse(pattern.directed)
            self.assertTrue(dags)

    def test_oracle_and_arbitrary_independences(self):
        rng = random.Random(44)
        for case in range(80):
            n = 4 if case < 60 else 5
            graph = skeleton(n, [edge for edge in it.combinations(range(n), 2) if rng.random() < .6])
            independent = []
            for i, j in it.combinations(range(n), 2):
                if not graph.has_edge(i, j):
                    rest = sorted(set(range(n)) - {i, j})
                    sets = [z for size in range(n - 1) for z in it.combinations(rest, size) if rng.random() < .5]
                    independent.extend((i, j, z) for z in sets or [()])
            pattern, realized = orient(graph, independent)
            variables = triples(graph)
            expected = {signature(dag, variables) for dag in all_dags(graph)
                        if pattern.directed <= set(dag.edges()) and all(
                            not value or t in pattern.ambiguous
                            or {(t[0], t[1]), (t[2], t[1])} <= pattern.directed
                            for t, value in assignment(dag, variables).items())}
            actual = [signature(dag, variables) for dag in realized]
            self.assertEqual(set(actual), expected, (case, graph.edges(), independent))
            self.assertEqual(len(actual), len(set(actual)))
            self.assertEqual({signature(dag, variables) for dag in representatives(pattern)}, expected)
            for i, k, j in pattern.ambiguous:
                self.assertTrue({tuple(sorted((i, k))), tuple(sorted((j, k)))} <= pattern.undirected)


if __name__ == "__main__":
    unittest.main()
