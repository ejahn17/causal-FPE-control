import unittest
from itertools import permutations

from experiments.summaries import operating_points, pr_auc


class SummaryTests(unittest.TestCase):
    def test_auc_ties_duplicates_and_input_order(self):
        points = [dict(recall=.2, precision=.9), dict(recall=.2, precision=.7),
                  dict(recall=.8, precision=.6)]
        expected = .2 * .85 + .6 * .75
        for order in permutations(points):
            self.assertAlmostEqual(pr_auc(order), expected)
            self.assertAlmostEqual(pr_auc([*order, points[0]]), expected)
        self.assertEqual(pr_auc([]), 0.)

    def test_operating_points_keep_failed_attempts_and_separate_settings(self):
        def record(alpha, precision=.8, recall=.4, status='ok'):
            return dict(algorithm='mcpc', params={'alpha': alpha}, status=status,
                        metrics=dict(precision=precision, recall=recall))
        points = {p['value']: p for p in operating_points([
            record(.1), record(.1, 1., .8), record(.1, status='algo_timeout'),
            record(.2), record(.3, status='metric_timeout')])}
        self.assertAlmostEqual(points[.1]['precision'], .9)
        self.assertAlmostEqual(points[.1]['recall'], .6)
        self.assertEqual((points[.1]['contributing'], points[.1]['attempted']), (2, 3))
        self.assertEqual(points[.2]['attempted'], 1)
        self.assertEqual(points[.1]['failed'], 1)
        self.assertEqual((points[.3]['contributing'], points[.3]['failed']), (0, 1))
        self.assertIsNone(points[.3]['precision'])
        self.assertIsNone(points[.3]['recall'])
        self.assertEqual(pr_auc([points[.3]]), 0.)

    def test_unequal_algorithm_quotas_do_not_reweight_means(self):
        rows = [dict(algorithm=a, params={'alpha': .1}, status='ok',
                     metrics=dict(precision=p, recall=1-p))
                for a, count, p in [('mcpc', 100, .8), ('pc', 1000, .4)] for _ in range(count)]
        points = {p['algorithm']: p for p in operating_points(rows)}
        for algorithm, count, precision in [('mcpc', 100, .8), ('pc', 1000, .4)]:
            self.assertEqual(points[algorithm]['contributing'], count)
            self.assertEqual(points[algorithm]['attempted'], count)
            self.assertAlmostEqual(points[algorithm]['precision'], precision)


if __name__ == '__main__':
    unittest.main()
