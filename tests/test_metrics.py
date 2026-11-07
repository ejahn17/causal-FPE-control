import unittest
from itertools import combinations, product

import networkx as nx
import numpy as np

from mcpc.baselines import InvalidGraph
from mcpc.graph import Pattern, cpdag
from mcpc.metrics import _aligned_member, evaluate, shd, similarity


def dag(n, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph


def tiny_dags(n):
    pairs = list(combinations(range(n), 2))
    for states in product(range(3), repeat=len(pairs)):
        graph = dag(n, [edge if state == 1 else edge[::-1]
                        for edge, state in zip(pairs, states) if state])
        if nx.is_directed_acyclic_graph(graph):
            yield graph


def independences(graph):
    result = set()
    for i, j in combinations(graph, 2):
        others = sorted(set(graph) - {i, j})
        for bits in product((False, True), repeat=len(others)):
            conditioned = frozenset(v for v, keep in zip(others, bits) if keep)
            if nx.is_d_separator(graph, {i}, {j}, set(conditioned)):
                result.add((i, j, conditioned))
    return result


class MetricTests(unittest.TestCase):
    def test_exact_tiny_graph_reference_bounds(self):
        # Exhaustive DAG/CI reference is independent of the deletion heuristic.
        graphs = list(tiny_dags(3))
        cis = [independences(g) for g in graphs]
        for a, estimate in enumerate(graphs):
            for b, truth in enumerate(graphs):
                required = cis[a] | cis[b]
                exact = max(g.number_of_edges() for g, statements in zip(graphs, cis)
                            if required <= statements)
                measured = evaluate(cpdag(estimate), [estimate], truth)
                self.assertGreaterEqual(measured["fd"], estimate.number_of_edges() - exact)
                self.assertGreaterEqual(measured["fn"], truth.number_of_edges() - exact)
                self.assertEqual(measured["fd"] - measured["fn"],
                                 estimate.number_of_edges() - truth.number_of_edges())
                self.assertAlmostEqual(measured["precision"], 1 - measured["fdr"])
                self.assertAlmostEqual(measured["recall"], 1 - measured["fnr"])

    def test_empty_complete_and_truth(self):
        truth = dag(6, [(0, 4), (1, 4), (2, 0), (2, 3), (4, 3), (5, 3)])
        cases = [(dag(6, []), 0, 6),
                 (dag(6, combinations(range(6), 2)), 9, 0), (truth, 0, 0)]
        for estimate, fd, fn in cases:
            metrics = evaluate(cpdag(estimate), [estimate], truth)
            self.assertEqual((metrics["fd"], metrics["fn"]), (fd, fn))
        self.assertEqual(evaluate(cpdag(truth), [truth], truth)["shd"], 0)

    def test_alignment_remains_markov_equivalent(self):
        rng = np.random.default_rng(2)
        for _ in range(30):
            order = rng.permutation(6).tolist()
            graph = dag(6, [(order[i], order[j]) for i, j in combinations(range(6), 2)
                            if rng.random() < .5])
            ranks = {node: i for i, node in enumerate(rng.permutation(6).tolist())}
            aligned = _aligned_member(graph, ranks)
            self.assertTrue(nx.is_directed_acyclic_graph(aligned))
            self.assertEqual(cpdag(graph), cpdag(aligned))

    def test_cpdag_edge_states_and_mec_minimum(self):
        collider = dag(3, [(0, 1), (2, 1)])
        chain = dag(3, [(0, 1), (1, 2)])
        self.assertEqual(shd(cpdag(chain), cpdag(collider)), 2)
        self.assertEqual(shd(Pattern(2, {(0, 1)}), Pattern(2, {(1, 0)})), 1)
        pattern = Pattern(3, set(), {(0, 1), (1, 2)}, {(0, 1, 2)})
        for truth in (chain, collider):
            result = evaluate(pattern, iter([chain, collider]), truth)
            self.assertEqual((result["fd"], result["fn"], result["shd"]), (0, 0, 0))

    def test_moral_marriages_repeat_until_separated(self):
        estimate = dag(4, [(0, 2), (1, 2), (0, 3), (1, 3)])
        truth = dag(4, [(0, 2), (2, 3), (3, 1)])
        rho = similarity(estimate, truth)
        self.assertGreaterEqual(rho, 0)
        self.assertLessEqual(rho, 3)

    def test_invalid_output_retains_structural_metrics(self):
        with self.assertRaises(InvalidGraph) as caught:
            evaluate(Pattern(3, {(0, 1), (1, 2), (2, 0)}), [], dag(3, []))
        self.assertEqual(caught.exception.metrics["rank"], 3)
        self.assertIsNone(caught.exception.metrics["fd"])


if __name__ == "__main__":
    unittest.main()
