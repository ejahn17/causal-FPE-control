import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from experiments import chambers, examples, faithfulness, pr, scalability
from experiments.models import covariance, example_model, materialize, random_model


def conditional_covariance(sigma, i, j, z):
    if not z:
        return sigma[i, j]
    return sigma[i, j] - sigma[i, z] @ np.linalg.solve(sigma[np.ix_(z, z)], sigma[z, j])


class ModelTests(unittest.TestCase):
    def test_example_constraints(self):
        constraints = {'fig4': [(0, 1, [])],
                       'fig5': [(0, 2, []), (0, 3, [])],
                       'fig6': [(0, 2, [3])]}
        for name, tests in constraints.items():
            for seed in (0, 1, 2):
                spec = dict(example=name, model_seed=seed, weight_range=(.5, 2.))
                weights, variances, labels, edges = example_model(spec)
                sigma = covariance(weights, variances)
                for i, j, z in tests:
                    self.assertLess(abs(conditional_covariance(sigma, i, j, z)), 1e-7)
                self.assertGreater(np.linalg.eigvalsh(sigma).min(), 0)

    def test_cached_sem_and_data_reproducible(self):
        spec = dict(kind='random', n_nodes=8, edge_probability=.3,
                    model_seed=12, data_seed=34, weight_range=(.5, 2.), n_samples=40)
        with tempfile.TemporaryDirectory() as directory:
            x, graph, info = materialize(spec, directory)
            again, graph2, info2 = materialize(spec, directory)
            np.testing.assert_array_equal(x, again)
            self.assertEqual(set(graph.edges), set(graph2.edges))
            self.assertTrue(nx.is_directed_acyclic_graph(graph))
            with np.load(info['model_path']) as model:
                np.testing.assert_allclose(covariance(model['weights'], model['variances']),
                                           model['covariance'], atol=1e-12)
                np.testing.assert_allclose(np.diag(model['covariance']), 1)
            short, _, short_info = materialize({**spec, 'n_samples': 10}, directory)
            np.testing.assert_array_equal(short, x[:10])
            self.assertEqual(info['model_key'], short_info['model_key'])
            self.assertEqual(len(list(Path(directory).glob('*.npz'))), 1)

    def test_full_grids(self):
        expected = {faithfulness: 1000, examples: 12000,
                    pr: 10500, scalability: 300, chambers: 113}
        for module, count in expected.items():
            tasks = list(module.tasks())
            self.assertEqual(len(tasks), count)
            self.assertEqual(len({task['id'] for task in tasks}), count)

    def test_cache_version_and_unknown_kind(self):
        spec = dict(kind='random', n_nodes=3, edge_probability=.3,
                    model_seed=12, data_seed=34, weight_range=(.5, 2.), n_samples=10)
        with tempfile.TemporaryDirectory() as directory:
            _, _, first = materialize(spec, directory)
            with patch('experiments.models.SAMPLER_VERSION', 99):
                _, _, second = materialize(spec, directory)
            self.assertNotEqual(first['model_key'], second['model_key'])
            with self.assertRaisesRegex(ValueError, 'Unknown data kind'):
                materialize({**spec, 'kind': 'typo'}, directory)

    def test_faithfulness_nested_conditions(self):
        spec = dict(kind='random', n_nodes=4, edge_probability=.8,
                    model_seed=9, data_seed=10, n_samples=100, weight_range=(.5, 2.))
        with tempfile.TemporaryDirectory() as directory:
            x, graph, _ = materialize(spec, directory)
            result = faithfulness.evaluate(x, graph, dict(alpha=.1, sample_sizes=(50, 100)))
        for row in result['rows']:
            self.assertGreaterEqual(row['faithfulness'], row['adjacency_faithfulness'])
            self.assertGreaterEqual(row['adjacency_faithfulness'], row['triangle_faithfulness'])
            self.assertGreaterEqual(row['m1'], row['m2'])
            self.assertGreaterEqual(row['m2'], row['m3'])

    def test_chambers_refinement(self):
        records = []
        for task in chambers.tasks():
            if task['algorithm'] == 'ges':
                value = task['params']['penalty']
                records.append({**task, 'pattern': {}, 'metrics': {'fn': 0 if value < .6 else 5}})
        proposals = chambers.next_points(records)
        self.assertEqual(len(proposals), 1)
        self.assertGreater(proposals[0]['params']['penalty'], .5)
        self.assertLess(proposals[0]['params']['penalty'], .6)
        for record in records:
            value = record['params']['penalty']
            record['metrics'] = None
            record['pattern'] = {'directed': [[0, 1], [1, 2], [2, 3]] if value < .6 else [],
                                 'undirected': []}
        self.assertEqual(len(chambers.next_points(records)), 1)
        for task in chambers.tasks():
            self.assertEqual(task['metric_timeout'], 600 if task['algorithm'] == 'mcpc' else 120)


if __name__ == '__main__':
    unittest.main()
