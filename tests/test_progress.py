import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from experiments.progress import (Checkpoint, TERMINAL, identity, model_progress)


def task(instance=0, setting=0, algorithm='mcpc', draw=0):
    return dict(id=f'pr_{instance}_{draw}_{setting}_{algorithm}', algorithm=algorithm,
                data_spec=dict(kind='random', n_nodes=10, model_seed=instance + 10,
                               data_seed=100 + draw, n_samples=100 + setting),
                params=dict(alpha=.1), group=dict(experiment='pr', instance=instance, draw=draw),
                algorithm_timeout=1, metric_timeout=1)


def outcome(task, status='ok'):
    return dict(task, status=status, code='execution', runtime=.1)


class ModelProgressTests(unittest.TestCase):
    def test_started_is_not_completed_and_repeated_datasets_are_one_model(self):
        tasks = [task(instance=i, setting=s, draw=d) for i in range(2)
                 for s in range(2) for d in range(2)]
        row, = model_progress(tasks, [outcome(tasks[0]), outcome(tasks[4], 'metric_timeout')])
        self.assertEqual((row['target_models'], row['started_models'], row['completed_models'],
                          row['fully_scored_models'], row['expected_fits'], row['completed_fits']),
                         (2, 2, 0, 0, 8, 2))
        records = [outcome(t) for t in tasks[:4]] + [outcome(t, 'invalid') for t in tasks[4:]]
        row, = model_progress(tasks, records)
        self.assertEqual((row['completed_models'], row['fully_scored_models']), (2, 1))

    def test_only_measured_terminal_outcomes_complete_models(self):
        statuses = sorted(TERMINAL) + ['error', 'crashed', 'preempted', 'data_timeout']
        tasks = [task(instance=i) for i in range(len(statuses))]
        records = [outcome(t, s) for t, s in zip(tasks, statuses)]
        row, = model_progress(tasks, records)
        self.assertEqual((row['started_models'], row['completed_models'], row['completed_fits']),
                         (10, 6, 6))
        self.assertEqual(row['fully_scored_models'], 1)
        self.assertTrue(all(row[status] == 1 for status in statuses))
        records += [outcome(tasks[-1])]
        row, = model_progress(tasks, records)
        self.assertEqual((row['completed_models'], row['data_timeout'], row['ok']), (7, 0, 2))

    def test_algorithms_have_separate_model_targets(self):
        tasks = [task(algorithm=a) for a in ('mcpc', 'pc', 'cpc', 'ges')]
        rows = model_progress(tasks, [outcome(t) for t in tasks])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['target_models'] == r['completed_models'] == 1 for r in rows))



class CheckpointTests(unittest.TestCase):
    def test_snapshot_and_replacements_preserve_exact_task_identity(self):
        tasks = [task(setting=i) for i in range(2)]
        old = outcome(tasks[0], 'preempted')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Checkpoint(tasks, [old], directory)
            checkpoint.add(outcome(tasks[0]))
            checkpoint.add(outcome(tasks[1], 'invalid'))
            with patch('experiments.progress.identity', side_effect=AssertionError('rehash')):
                result = checkpoint.snapshot('complete')
            saved = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(saved, result)
            self.assertEqual(saved['totals']['completed_models'], 1)
            self.assertEqual(saved['rows'][0]['preempted'], 0)
            self.assertEqual(set(checkpoint.records), {identity(t) for t in tasks})
            self.assertFalse(list(Path(directory).glob('.progress*')))

    def test_errors_are_saved_before_raise_and_interruptions_do_not_raise(self):
        item = task()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Checkpoint([item], [], directory)
            checkpoint.add(outcome(item, 'preempted'))
            with self.assertRaisesRegex(RuntimeError, 'execution error'):
                checkpoint.add(outcome(item, 'error'))
            self.assertEqual(checkpoint.records[identity(item)]['status'], 'error')
            saved = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual((saved['state'], saved['totals']['completed_fits']), ('needs_attention', 0))
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                checkpoint.snapshot('complete')
            checkpoint.snapshot('interrupted')
            self.assertEqual(json.loads((Path(directory) / 'progress.json').read_text())['state'],
                             'interrupted')

    def test_reports_and_plots_use_elapsed_intervals(self):
        item = task()
        with tempfile.TemporaryDirectory() as directory, patch('experiments.progress.time.monotonic') as clock:
            clock.return_value = 0
            checkpoint = Checkpoint([item], [], directory)
            with patch('experiments.plots.render') as render:
                clock.return_value = 29
                checkpoint.add(outcome(item))
                self.assertFalse((Path(directory) / 'progress.json').exists())
                clock.return_value = 30
                checkpoint.add(outcome(item))
                render.assert_not_called()
                clock.return_value = 300
                checkpoint.add(outcome(item))
                checkpoint.close()
                render.assert_called_once_with('pr', list(checkpoint.records.values()),
                                               Path(directory))

    def test_background_render_does_not_block_callbacks_and_final_waits(self):
        tasks = [task(setting=i) for i in range(3)]
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        batches, errors, active, maximum = [], [], 0, 0
        def render(name, records, path):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            batches.append(list(records))
            started.set()
            release.wait(5)
            active -= 1
        with tempfile.TemporaryDirectory() as directory, patch('experiments.plots.render', side_effect=render):
            checkpoint = Checkpoint(tasks, [], directory, plot_every=1)
            checkpoint.last_plot = float('-inf')
            def add():
                try:
                    checkpoint.add(outcome(tasks[0]))
                except BaseException as error:
                    errors.append(error)
                finally:
                    finished.set()
            caller = threading.Thread(target=add)
            caller.start()
            try:
                self.assertTrue(started.wait(1))
                self.assertTrue(finished.wait(1), 'render blocked the completion callback')
                self.assertFalse(errors)
                checkpoint.add(outcome(tasks[1]))
                with self.assertRaisesRegex(RuntimeError, 'execution error'):
                    checkpoint.add(outcome(tasks[2], 'error'))
                self.assertEqual(len(batches), 1)  # No second pending plot.
                def final_snapshot():
                    try:
                        checkpoint.snapshot('needs_attention', plots=True)
                    except BaseException as error:
                        errors.append(error)
                final = threading.Thread(target=final_snapshot)
                final.start()
                final.join(.05)
                self.assertTrue(final.is_alive(), 'final snapshot failed to wait for rendering')
            finally:
                release.set()
                caller.join(2)
                if 'final' in locals():
                    final.join(2)
                checkpoint.close()
            self.assertFalse(errors)
            self.assertEqual([len(batch) for batch in batches], [1, 3])
            self.assertEqual(maximum, 1)
            self.assertIsNone(checkpoint.executor)

    def test_background_render_error_surfaces_after_storing_next_record(self):
        tasks = [task(setting=i) for i in range(2)]
        with tempfile.TemporaryDirectory() as directory, patch(
                'experiments.plots.render', side_effect=ValueError('figure failed')):
            checkpoint = Checkpoint(tasks, [], directory, plot_every=1)
            checkpoint.last_plot = float('-inf')
            checkpoint.add(outcome(tasks[0]))
            self.assertIsInstance(checkpoint.pending_plot.exception(timeout=2), ValueError)
            with self.assertRaisesRegex(ValueError, 'figure failed'):
                checkpoint.add(outcome(tasks[1]))
            self.assertEqual(len(checkpoint.records), 2)
            checkpoint.snapshot('needs_attention')
            self.assertIsNone(checkpoint.executor)

    def test_final_snapshot_surfaces_background_error_and_closes_thread(self):
        item = task()
        with tempfile.TemporaryDirectory() as directory, patch(
                'experiments.plots.render', side_effect=ValueError('figure failed')):
            checkpoint = Checkpoint([item], [], directory, plot_every=1)
            checkpoint.last_plot = float('-inf')
            checkpoint.add(outcome(item))
            with self.assertRaisesRegex(ValueError, 'figure failed'):
                checkpoint.snapshot('complete', plots=True)
            self.assertIsNone(checkpoint.executor)


if __name__ == '__main__':
    unittest.main()
