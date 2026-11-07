import unittest
from itertools import combinations, product
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

from mcpc.baselines import InvalidGraph, _decode, fit, local_score_BIC_from_cov
from mcpc.graph import cpdag, triples


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.default_rng(17).normal(size=(600, 4))
        cls.data[:, 1] += cls.data[:, 0]
        cls.data[:, 2] += cls.data[:, 1]

    def test_all_methods_return_valid_patterns(self):
        for method in ("pc", "cpc", "ges"):
            result = fit(self.data, method, {"alpha": .01, "penalty": .5})
            dags = list(result.representatives)
            self.assertTrue(dags)
            for graph in dags:
                self.assertTrue(nx.is_directed_acyclic_graph(graph))
                self.assertEqual(set(graph.to_undirected().edges()),
                                 set(result.pattern.skeleton().edges()))

    def test_alpha_zero_is_lower_limit_and_keeps_zero_p_edges(self):
        from causallearn.utils.cit import CIT
        rng = np.random.default_rng(100)
        data = rng.normal(size=(500, 3))
        data[:, 1] = data[:, 0] + .01 * data[:, 1]
        self.assertEqual(CIT(data, 'fisherz')(0, 1, []), 0.)
        lower_limit = float(np.nextafter(0., 1.))
        for method in ('pc', 'cpc'):
            params = {'alpha': 0.}
            zero = fit(data, method, params)
            tiny = fit(data, method, {'alpha': lower_limit})
            self.assertEqual(zero.pattern, tiny.pattern)
            self.assertTrue(zero.pattern.skeleton().has_edge(0, 1))
            self.assertTrue(list(zero.representatives))
            self.assertEqual(params, {'alpha': 0.})
            self.assertEqual(zero.diagnostics['alpha_requested'], 0.)
            self.assertEqual(zero.diagnostics['alpha_effective'], lower_limit)

    def test_pc_cpc_reject_invalid_alpha(self):
        for method in ('pc', 'cpc'):
            for alpha in (-1e-6, 1., 2., np.nan, np.inf, -np.inf):
                with self.assertRaisesRegex(ValueError, r'alpha.*\[0, 1\)'):
                    fit(self.data, method, {'alpha': alpha})

    def test_entire_baseline_parameter_grids_on_tiny_data(self):
        from experiments import pr, chambers
        alpha_grid = sorted(set(pr.PC_ALPHA_GRID) | set(chambers.PC_ALPHA_GRID))
        penalty_grid = sorted(set(pr.GES_PENALTY_GRID) | set(chambers.GES_PENALTY_GRID))
        for method in ('pc', 'cpc', 'ges'):
            grid, field = (penalty_grid, 'penalty') if method == 'ges' else (alpha_grid, 'alpha')
            for value in grid:
                with self.subTest(method=method, value=value):
                    result = fit(self.data[:80, :3], method, {field: value})
                    self.assertTrue(list(result.representatives))

    def test_cpc_exhaustive_lazy_family(self):
        matrix = np.zeros((4, 4), dtype=int)
        matrix[0, 1:] = matrix[1:, 0] = -1
        cg = SimpleNamespace(G=SimpleNamespace(graph=matrix),
                             definite_UC=[], definite_non_UC=[])
        with patch("causallearn.search.ConstraintBased.PC.pc", return_value=cg):
            result = fit(self.data, "cpc", {"alpha": .1})
        self.assertNotIsInstance(result.representatives, list)
        classes = [cpdag(g).to_dict() for g in result.representatives]
        self.assertEqual(len(classes), 5)  # noncollider, three pairs, all three collider triples
        self.assertEqual(len({str(p) for p in classes}), 5)
        self.assertEqual(len(result.pattern.ambiguous), 3)

    def test_cpc_family_matches_independent_edge_enumeration(self):
        rng = np.random.default_rng(9)
        for _ in range(20):
            skeleton = nx.Graph()
            skeleton.add_nodes_from(range(4))
            skeleton.add_edges_from(e for e in combinations(range(4), 2) if rng.random() < .6)
            source = nx.DiGraph()
            source.add_nodes_from(range(4))
            source.add_edges_from(skeleton.edges())
            fixed = {t: source.has_edge(t[0], t[1]) and source.has_edge(t[2], t[1])
                     for t in triples(skeleton) if rng.random() < .6}
            matrix = np.zeros((4, 4), dtype=int)
            for i, j in skeleton.edges():
                matrix[i, j] = matrix[j, i] = -1
            cg = SimpleNamespace(G=SimpleNamespace(graph=matrix),
                                 definite_UC=[t for t, value in fixed.items() if value],
                                 definite_non_UC=[t for t, value in fixed.items() if not value])
            with patch('causallearn.search.ConstraintBased.PC.pc', return_value=cg):
                result = fit(self.data, 'cpc', {})
            obtained = {str(cpdag(g).to_dict()) for g in result.representatives}
            expected = set()
            edges = list(skeleton.edges())
            for bits in product((False, True), repeat=len(edges)):
                candidate = nx.DiGraph()
                candidate.add_nodes_from(range(4))
                candidate.add_edges_from(e if forward else e[::-1] for e, forward in zip(edges, bits))
                if nx.is_directed_acyclic_graph(candidate) and all(
                        (candidate.has_edge(i, k) and candidate.has_edge(j, k)) == value
                        for (i, k, j), value in fixed.items()):
                    expected.add(str(cpdag(candidate).to_dict()))
            self.assertEqual(obtained, expected)

    def test_bic_formula_and_singular_covariance(self):
        covariance = np.array([[2., 1.], [1., 3.]])
        expected = -.5 * 100 * (1 + np.log(2 - 1 / 3)) - np.log(100)
        self.assertAlmostEqual(local_score_BIC_from_cov((covariance, 100), 0, [1]), expected)
        self.assertTrue(np.isfinite(local_score_BIC_from_cov((np.ones((3, 3)), 100), 0, [1, 2])))

    def test_ges_patch_restored_after_success_and_failure(self):
        import causallearn.search.ScoreBased.GES as module
        original = module.local_score_BIC_from_cov
        fit(self.data, "ges", {"penalty": .5})
        self.assertIs(module.local_score_BIC_from_cov, original)
        with patch.object(module, "ges", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                fit(self.data, "ges", {})
        self.assertIs(module.local_score_BIC_from_cov, original)

    def test_ges_penalty_sweep_is_forwarded(self):
        import causallearn.search.ScoreBased.GES as module
        output = {"G": SimpleNamespace(graph=np.zeros((4, 4), dtype=int))}
        with patch.object(module, "ges", return_value=output) as search:
            for penalty in (0., .5, 1000.):
                fit(self.data, "ges", {"penalty": penalty})
                self.assertEqual(search.call_args.kwargs["lambda_value"], penalty)

    def test_conflicting_or_nonextendable_outputs_are_invalid(self):
        with self.assertRaises(InvalidGraph) as caught:
            _decode(np.array([[0, 1], [1, 0]]))
        self.assertEqual(caught.exception.metrics['rank'], 1)
        self.assertIsNotNone(caught.exception.pattern)
        matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        cg = SimpleNamespace(G=SimpleNamespace(graph=matrix))
        with patch("causallearn.search.ConstraintBased.PC.pc", return_value=cg):
            with self.assertRaises(InvalidGraph) as caught:
                fit(self.data[:, :3], "pc", {})
        self.assertEqual(caught.exception.metrics['rank'], 3)
        self.assertIsNotNone(caught.exception.pattern)


if __name__ == "__main__":
    unittest.main()
