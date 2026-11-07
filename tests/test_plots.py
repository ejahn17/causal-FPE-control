import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib.pyplot as plt

from experiments.plots import (atomic_path, comparison, edge_kind, precision_recall, render,
                               save, truth_edge_reference)
from mcpc.graph import Pattern


class PlotTests(unittest.TestCase):
    def test_atomic_outputs_keep_previous_file_on_interrupt_and_use_unique_names(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / 'figure.png'
            destination.write_bytes(b'previous')
            with self.assertRaises(KeyboardInterrupt):
                with atomic_path(destination) as temporary:
                    temporary.write_bytes(b'incomplete')
                    raise KeyboardInterrupt
            self.assertEqual(destination.read_bytes(), b'previous')
            with atomic_path(destination) as first:
                first.write_bytes(b'first complete')
                with atomic_path(destination) as second:
                    self.assertNotEqual(first, second)
                    second.write_bytes(b'second complete')
                self.assertEqual(destination.read_bytes(), b'second complete')
            self.assertEqual(destination.read_bytes(), b'first complete')
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_failed_figure_write_leaves_previous_image_and_closes_figure(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / 'cpdag_mcpc.png'
            destination.write_bytes(b'previous')
            fig = plt.figure()
            def interrupted(path, **kwargs):
                path.write_bytes(b'partial image')
                raise KeyboardInterrupt
            with patch.object(fig, 'savefig', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    save(fig, directory, 'cpdag_mcpc', formats=('png',))
            self.assertFalse(plt.fignum_exists(fig.number))
            self.assertEqual(destination.read_bytes(), b'previous')
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_truth_reference_counts_datasets_once_and_includes_failed_fits(self):
        first = dict(data_spec={'model_seed': 1}, group={'n_samples': 100},
                     truth=[(0, 1), (1, 2)], status='ok')
        second = dict(data_spec={'model_seed': 2}, group={'n_samples': 100},
                      truth=[(0, 1), (1, 2), (2, 3), (3, 4)], status='invalid')
        self.assertEqual(truth_edge_reference([first, first, first, second]),
                         ([100], [3.], 'Mean true adjacencies'))

    def test_example_adjacencies_panel_has_constant_ground_truth_line(self):
        records = [dict(algorithm='mcpc', status='ok', data_spec={'n_samples': n},
                        group={'n_samples': n}, truth=[(0, 1), (1, 2)],
                        metrics=dict(fd=0, fdr=0., rank=1, n_amb=0)) for n in (100, 1000)]
        with patch('experiments.plots.save') as save:
            comparison(records, '.', 'fig4')
        fig = save.call_args.args[0]
        try:
            axes = {ax.get_ylabel(): ax for ax in fig.axes}
            self.assertEqual(set(axes), {'FD', 'FDR', 'Adjacencies', 'Ambiguous triples'})
            line, = [line for line in axes['Adjacencies'].lines if line.get_label() == 'True adjacencies = 2']
            self.assertEqual(list(line.get_ydata()), [2., 2.])
        finally:
            plt.close(fig)

    def test_only_reversed_compelled_arrows_are_marked_wrong(self):
        truth = Pattern(4, {(0, 1)}, {(1, 2)}, set())
        self.assertEqual(edge_kind((1, 0), True, truth), 'reversed')
        self.assertEqual(edge_kind((0, 1), True, truth), 'correct')
        self.assertEqual(edge_kind((0, 1), False, truth), 'correct')
        for edge in ((1, 2), (2, 1)):
            self.assertEqual(edge_kind(edge, True, truth), 'correct')
        self.assertEqual(edge_kind((2, 3), True, truth), 'extra')

    def test_pr_keeps_paper_output_path(self):
        records = [dict(data_spec={'n_nodes': 10})]
        with patch('experiments.plots.render_group') as draw:
            render('pr', records, '/result')
        draw.assert_called_once_with('pr', records, Path('/result/n10'))

    def test_failure_only_partial_pr_reports_unavailable_auc_for_each_algorithm(self):
        records = [dict(algorithm='mcpc', params={'alpha': .1}, status='metric_timeout', metrics=None)]
        with TemporaryDirectory() as directory:
            with patch('experiments.plots.save') as save, patch('experiments.plots.write_table') as table:
                precision_recall(records, directory)
            plt.close(save.call_args.args[0])
        areas = next(call.args[2] for call in table.call_args_list if call.args[1] == 'pr_auc')
        self.assertEqual(len(areas), 4)
        self.assertTrue(all(row['pr_auc'] is None and row['points'] == 0 for row in areas))

    def test_scalability_counts_completed_algorithms_when_metrics_time_out(self):
        def record(algorithm, status, runtime, stage='metric'):
            return dict(algorithm=algorithm, status=status, runtime=runtime, stage=stage,
                        group=dict(experiment='scalability', n_nodes=10, degree=3),
                        params=dict(m=1, ugl_penalty=.2) if algorithm == 'mcpc' else {})
        records = [record('mcpc', 'ok', 2), record('mcpc', 'metric_timeout', 8),
                   record('mcpc', 'algo_timeout', 300, 'algorithm')]
        with TemporaryDirectory() as directory:
            render('scalability', records, directory)
            root = Path(directory)
            with (root / 'scalability.csv').open() as stream:
                runtime, = list(csv.DictReader(stream))
            self.assertEqual(float(runtime['median_seconds']), 5.)
            self.assertEqual((runtime['completed_algorithms'], runtime['attempted']), ('2', '3'))
            self.assertFalse({'algorithm', 'm', 'ugl_penalty'} & runtime.keys())
            self.assertFalse(any((root / name).exists() for name in ('metrics.csv', 'summary.csv', 'summary.md')))


if __name__ == '__main__':
    unittest.main()
