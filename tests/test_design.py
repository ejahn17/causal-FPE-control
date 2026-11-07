import hashlib
import json
import unittest
from collections import Counter, defaultdict
from itertools import groupby

from experiments import pr, scalability


def fingerprint(tasks):
    return hashlib.sha256(json.dumps(tasks, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class DesignTests(unittest.TestCase):
    def test_publication_tasks_preserve_completed_payloads_and_order(self):
        # Snapshots from the completed paper experiments, before publication pruning.
        tasks = list(pr.tasks())
        self.assertEqual(fingerprint(sorted(tasks, key=lambda t: t['id'])),
                         '1d15bbb266fb76bb6cd903cb0c8911a476f3933ce6f1feb2966acf050f1ac04a')
        self.assertEqual(fingerprint(tasks),
                         'e855c828d2f5aa8d2a4922bd4488d15fa7641be37fd2fa6741308054779e4860')
        self.assertEqual(fingerprint(list(scalability.tasks())),
                         '6ea40c7c4829bd9458f6fe7ae8f28704c7cdc25f39093d82a47bd9c4168a6712')

    def test_pr_has_one_shared_dataset_per_model_and_complete_parameter_rounds(self):
        tasks = list(pr.tasks())
        self.assertEqual(Counter(t['algorithm'] for t in tasks),
                         dict(mcpc=2200, pc=2900, cpc=2900, ges=2500))
        self.assertEqual(len({t['id'] for t in tasks}), 10500)
        specifications, instances = {}, defaultdict(set)
        for task in tasks:
            instance, spec = task['group']['instance'], task['data_spec']
            self.assertEqual(specifications.setdefault(instance, spec), spec)
            self.assertEqual((spec['n_nodes'], spec['n_samples'], spec['edge_probability']),
                             (10, 1000, 3/9))
            instances[task['algorithm']].add(instance)
        self.assertEqual(len({s['model_seed'] for s in specifications.values()}), 100)
        self.assertTrue(all(values == set(range(100)) for values in instances.values()))
        blocks = [(instance, list(rows)) for instance, rows in
                  groupby(tasks, key=lambda t: t['group']['instance'])]
        self.assertEqual([i for i, _ in blocks], list(range(100)))
        for _, rows in blocks:
            self.assertEqual(len(rows), 105)
            self.assertEqual([t['algorithm'] for t in rows[:8]], ['mcpc', 'pc', 'cpc', 'ges']*2)

    def test_scalability_has_ten_models_per_published_cell(self):
        tasks = list(scalability.tasks())
        self.assertEqual(len(tasks), 300)
        cells = Counter((t['group']['n_nodes'], t['group']['degree']) for t in tasks)
        self.assertEqual(set(cells), {(n, d) for n in (10, 12, 15, 18, 20, 25) for d in range(1, 6)})
        self.assertEqual(set(cells.values()), {10})
        self.assertTrue(all(t['algorithm'] == 'mcpc' and t['params']['m'] == 1
                            and t['params']['ugl_penalty'] == .2 for t in tasks))


if __name__ == '__main__':
    unittest.main()
