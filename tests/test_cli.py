import argparse
import json
import tempfile
import unittest
from pathlib import Path
from itertools import islice
from unittest.mock import patch

from experiments import __main__ as cli, chambers, pr, runner
from mcpc.graph import Pattern


def record(task, code='revision', status='ok', **extra):
    return dict(task, code=code, status=status, pattern=Pattern(3).to_dict(),
                metrics=dict(n_triples=3, n_amb=1),
                key=runner.digest({'task': task, 'code': code}), **extra)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(cli, 'code_digest', return_value='revision')
        patcher.start()
        self.addCleanup(patcher.stop)
        self.task = chambers._task('mcpc', .05, 'data.csv')

    def test_verified_reuse_preserves_original_execution(self):
        old = record(self.task, 'previous')
        self.assertEqual(cli.cache_plan([self.task], [old]), ([], [self.task]))
        old['validated_for'] = 'revision'
        cached, pending = cli.cache_plan([self.task], [old])
        self.assertEqual(pending, [])
        self.assertIs(cached[0], old)
        self.assertEqual(cached[0]['code'], 'previous')
        for changed in ({**self.task, 'params': {**self.task['params'], 'alpha': .1}},
                        {**self.task, 'data_spec': {**self.task['data_spec'], 'path': 'other.csv'}},
                        {**self.task, 'metric_timeout': 121}):
            self.assertEqual(cli.cache_plan([changed], [old]), ([], [changed]))
        self.assertEqual(cli.cache_plan([self.task], [old], current='other'), ([], [self.task]))
        forged = {**old, 'key': 'not-authentic'}
        self.assertEqual(cli.compatible_records([forged]), [])

    def test_current_retry_supersedes_verified_earlier_success(self):
        old = record(self.task, 'previous', validated_for='revision')
        for status in runner.RETRY:
            retry = record(self.task, status=status)
            for saved in ([old, retry], [retry, old]):
                self.assertEqual(cli.cache_plan([self.task], saved), ([], [self.task]))
        for status in cli.TERMINAL:
            done = record(self.task, status=status)
            self.assertEqual(cli.cache_plan([self.task], [done]), ([done], []))

    def test_chambers_selection_includes_only_requested_families(self):
        initial = list(chambers.tasks(data_path='data.csv'))
        extra = chambers._task('cpc', .123, 'data.csv')
        other = chambers._task('cpc', .123, 'other.csv')
        wrong = dict(extra, algorithm='unknown')
        saved = [record(t) for t in [*initial, extra, other, wrong]]
        selected = cli.select_records([*saved, saved[0]], initial, 'chambers')
        self.assertEqual(len(selected), len(initial)+1)
        self.assertEqual({r['algorithm'] for r in selected}, {'mcpc', 'pc', 'cpc', 'ges'})

    def test_cached_run_checks_environment_without_rewriting(self):
        old = record(self.task, 'previous', validated_for='revision')
        env = dict(code='previous', packages={'numpy': 'test'})
        with tempfile.TemporaryDirectory() as directory:
            path = runner.prepare_store(directory)
            runner.append_record(path, old)
            ep = Path(directory)/'environment.json'
            ep.write_text(json.dumps(env))
            original = path.read_bytes(), ep.read_bytes()
            with patch.object(cli, 'execute') as execute, patch.object(
                    cli, 'environment', return_value={**env, 'code': 'revision'}):
                self.assertEqual(cli.run([self.task], directory), [old])
                execute.assert_not_called()
            self.assertEqual((path.read_bytes(), ep.read_bytes()), original)
            with patch.object(cli, 'environment', return_value={**env, 'packages': {}}):
                with self.assertRaisesRegex(ValueError, 'environment changed'):
                    cli.run([self.task], directory)

    def test_pr_retries_only_unfinished_tasks(self):
        tasks = list(islice(pr.tasks(), 4))
        calls = []
        def execute(pending, directory, workers, on_record):
            calls.append(pending)
            for i, task in enumerate(pending):
                on_record(record(task, status='preempted' if len(calls) == 1 and i == 0 else 'ok'))
            return []
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(workers=2, plot_every=300)
            with patch.object(cli, 'execute', side_effect=execute), patch('experiments.plots.render'):
                cli.run_pr(tasks, [], Path(directory), args)
            value = json.loads((Path(directory)/'progress.json').read_text())
            self.assertEqual(value['state'], 'complete')
            self.assertEqual(value['totals']['completed_fits'], 4)
        self.assertEqual(calls, [tasks, tasks[:1]])

    def test_interruption_refreshes_durable_cleanup_outcomes(self):
        tasks = list(islice(pr.tasks(), 4))
        def execute(pending, directory, workers, on_record):
            path = runner.prepare_store(directory)
            first = record(pending[0])
            runner.append_record(path, first)
            on_record(first)
            runner.append_record(path, record(pending[1], status='preempted'))
            raise runner.RunInterrupted([], 15)
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(workers=2, plot_every=300)
            with patch.object(cli, 'execute', side_effect=execute) as run, patch('experiments.plots.render'):
                with self.assertRaises(runner.RunInterrupted):
                    cli.run_pr(tasks, [], Path(directory), args)
                self.assertEqual(run.call_count, 1)
            value = json.loads((Path(directory)/'progress.json').read_text())
            self.assertEqual(value['state'], 'interrupted')
            self.assertEqual(sum(row['preempted'] for row in value['rows']), 1)
            self.assertEqual(value['totals']['completed_fits'], 1)

    def test_callback_error_refreshes_durable_cleanup_outcomes(self):
        tasks = list(islice(pr.tasks(), 4))
        def execute(pending, directory, workers, on_record):
            path = runner.prepare_store(directory)
            failed = record(pending[0], status='error')
            runner.append_record(path, failed)
            try:
                on_record(failed)
            except RuntimeError:
                runner.append_record(path, record(pending[1], status='preempted'))
                raise
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(workers=2, plot_every=300)
            with patch.object(cli, 'execute', side_effect=execute), patch('experiments.plots.render'):
                with self.assertRaisesRegex(RuntimeError, 'execution error'):
                    cli.run_pr(tasks, [], Path(directory), args)
            value = json.loads((Path(directory)/'progress.json').read_text())
            self.assertEqual(value['state'], 'needs_attention')
            self.assertEqual(sum(row['error'] for row in value['rows']), 1)
            self.assertEqual(sum(row['preempted'] for row in value['rows']), 1)

    def test_pr_plot_only_reads_partial_saved_results_without_fitting(self):
        tasks = list(islice(pr.tasks(), 4))
        with tempfile.TemporaryDirectory() as directory:
            path = runner.prepare_store(Path(directory)/'pr')
            saved = record(tasks[0])
            runner.append_record(path, saved)
            before = path.read_bytes()
            args = argparse.Namespace(out_dir=Path(directory), data_path=None, plot_only=True)
            with patch.object(pr, 'tasks', return_value=tasks), patch.object(cli, 'render') as render, \
                    patch.object(cli, 'execute') as execute:
                cli.regular('pr', args)
                render.assert_called_once_with('pr', [json.loads(json.dumps(saved))], Path(directory)/'pr')
                execute.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_all_runs_only_publication_suites(self):
        with patch('sys.argv', ['experiments', 'all']), patch.object(cli, 'regular') as run:
            cli.main()
        self.assertEqual([call.args[0] for call in run.call_args_list],
                         ['faithfulness', 'examples', 'pr', 'scalability', 'chambers'])


if __name__ == '__main__':
    unittest.main()
