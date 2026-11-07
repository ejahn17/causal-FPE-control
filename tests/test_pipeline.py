import unittest
from itertools import combinations

import networkx as nx
import numpy as np

from mcpc import mcpc
from mcpc.ci import FisherZ, by_test, test_list
from mcpc.graph import cpdag


class CITests(unittest.TestCase):
    def test_fisher_z_matches_baseline(self):
        from causallearn.utils.cit import CIT
        data = np.random.default_rng(42).normal(size=(400, 5))
        data[:, 1] += 1.8 * data[:, 0]
        ours, reference = FisherZ(data), CIT(data, 'fisherz')
        for i, j in combinations(range(5), 2):
            rest = [k for k in range(5) if k not in (i, j)]
            for size in range(3):
                for conditioned in combinations(rest, size):
                    self.assertAlmostEqual(ours(i, j, conditioned),
                                           reference(i, j, list(conditioned)), places=13)

    def test_neighborhood_test_list_is_complete_without_duplicates(self):
        graph = nx.Graph([(0, 1), (1, 2), (2, 3), (1, 4)])
        for m in (1, 2):
            expected = set()
            for i, j in combinations(range(5), 2):
                for endpoint in (i, j):
                    neighborhood = sorted(set(nx.single_source_shortest_path_length(
                        graph, endpoint, cutoff=m)) - {i, j})
                    for size in range(len(neighborhood) + 1):
                        expected.update((i, j, z) for z in combinations(neighborhood, size))
            actual = list(test_list(graph, m))
            self.assertEqual(set(actual), expected)
            self.assertEqual(len(actual), len(expected))

    def test_by_boundary_and_pair_maximum(self):
        tests = [(0, 1, ()), (0, 1, (2,)), (0, 2, ()),
                 (0, 2, (1,)), (1, 2, ()), (1, 2, (0,))]
        values = dict(zip(tests, (.05, 0, .2, .05, .9, .2)))
        calls = []
        def pvalue(i, j, z):
            calls.append((i, j, z))
            return values[i, j, z]
        independent, info = by_test(lambda: iter(tests), pvalue, .55)
        self.assertEqual(calls, tests)  # The second pass must not repeat CI tests.
        self.assertEqual(info['selected_pairs'], 2)
        self.assertAlmostEqual(info['threshold'], .2)
        self.assertEqual(independent, [(1, 2, ())])  # Equality stays dependent.

    def test_by_zero_alpha_and_invalid_pvalues(self):
        tests = [(0, 1, ()), (0, 2, ())]
        independent, info = by_test(lambda: iter(tests), lambda i, j, z: j-1, 0)
        self.assertEqual(independent, [(0, 2, ())])
        self.assertEqual(info['selected_pairs'], 1)
        for invalid in (float('nan'), -.1, 1.1):
            with self.assertRaises(ValueError):
                by_test(lambda: iter(tests), lambda *args: invalid, .1)


class PipelineTests(unittest.TestCase):
    def test_oracle_chain_collider_and_empty_graph(self):
        for edges in ([(0, 1), (1, 2)], [(0, 1), (2, 1)], []):
            truth = nx.DiGraph()
            truth.add_nodes_from(range(3))
            truth.add_edges_from(edges)
            result = mcpc(np.zeros((10, 3)), ugl=lambda x: nx.complete_graph(3),
                          ci_test=lambda i, j, z: float(nx.is_d_separator(
                              truth, {i}, {j}, set(z))))
            self.assertEqual(result.pattern.to_dict(), cpdag(truth).to_dict())
            self.assertTrue(result.representatives)
            self.assertEqual(result.diagnostics['n_tests'], 6)

    def test_invalid_inputs_are_rejected(self):
        good = np.random.default_rng(3).normal(size=(20, 3))
        for settings in ({'m': 0}, {'m': 1.5}, {'alpha': -.1}, {'ugl_rule': 'and'}):
            with self.assertRaises(ValueError):
                mcpc(good, **settings)
        for data in (np.ones((20, 3)), np.array([[np.nan]]), np.zeros(4)):
            with self.assertRaises(ValueError):
                mcpc(data)


if __name__ == '__main__':
    unittest.main()
