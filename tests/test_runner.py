import json
import hashlib
import multiprocessing as mp
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from experiments import runner
from mcpc.algorithm import Result
from mcpc.graph import Pattern


def tiny_task(identifier='tiny'):
    return dict(id=identifier, algorithm='faithfulness',
                data_spec=dict(kind='random', n_nodes=3, edge_probability=.3,
                               model_seed=1, data_seed=2, n_samples=30, weight_range=(.5, 2.)),
                params=dict(alpha=.1, sample_sizes=[30]),
                group=dict(experiment='faithfulness', instance=0),
                algorithm_timeout=10, metric_timeout=10)


def controlled_worker(connection, cache):
    """Exercise parent cancellation with queued messages and an unresponsive child."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    task = connection.recv()
    partial = dict(task, pattern=Pattern(3).to_dict(), runtime=.125, metrics=None,
                   worker_pid=os.getpid())
    connection.send({'partial': partial})
    connection.send({'stage': 'metric', 'limit': 30})
    Path(cache, task['id'] + '.ready').write_text('ready')
    until = time.monotonic() + 10
    while len(list(Path(cache).glob('*.ready'))) < task['params'].get('wait_for', 1):
        if time.monotonic() > until:
            raise RuntimeError('Test workers did not become ready')
        time.sleep(.01)
    if task['params'].get('complete'):
        connection.send({'record': dict(partial, status='ok', stage='metric', metrics={})})
    if task['params'].get('signal'):
        os.kill(os.getppid(), task['params']['signal'])
    while True:
        time.sleep(1)


def sleeping_worker(connection, cache):
    from experiments import faithfulness
    evaluate = faithfulness.evaluate
    def delayed(data, truth, params):
        if params.get('pause'):
            time.sleep(1)
        return evaluate(data, truth, params)
    with patch.object(faithfulness, 'evaluate', side_effect=delayed):
        runner._worker(connection, cache)


class StoreTests(unittest.TestCase):
    def test_duplicate_records_keep_latest_and_internal_corruption_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = runner.prepare_store(directory)
            runner.append_record(path, {'key': 'a', 'status': 'error'})
            runner.append_record(path, {'key': 'a', 'status': 'ok'})
            self.assertEqual(runner.load_records(path)['a']['status'], 'ok')
            with path.open('a') as handle:
                handle.write('broken\n' + json.dumps({'key': 'b'}) + '\n')
            with self.assertRaises(json.JSONDecodeError):
                runner.load_records(path)

    def test_torn_tail_repaired_before_append(self):
        for ending in ('{"key":', '{"key":\n'):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as directory:
                path = runner.prepare_store(directory)
                runner.append_record(path, {'key': 'a'})
                with path.open('a') as handle:
                    handle.write(ending)
                self.assertEqual(set(runner.load_records(path)), {'a'})
                runner.prepare_store(directory)
                runner.append_record(path, {'key': 'b'})
                self.assertEqual(set(runner.load_records(path)), {'a', 'b'})

    def test_complete_final_line_without_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = runner.prepare_store(directory)
            path.write_text('{"key":"a"}')
            runner.prepare_store(directory)
            runner.append_record(path, {'key': 'b'})
            self.assertEqual(set(runner.load_records(path)), {'a', 'b'})


class WorkerTests(unittest.TestCase):
    def test_work_errors_and_stage_timeouts(self):
        with tempfile.TemporaryDirectory() as cache:
            task = tiny_task()
            broken = {**task, 'data_spec': {**task['data_spec'], 'kind': 'invalid'}}
            record = runner._work(broken, cache, lambda message: None)
            self.assertEqual((record['status'], record['stage']), ('error', 'data'))
            with patch('experiments.faithfulness.evaluate', side_effect=lambda *args: time.sleep(.1)):
                record = runner._work({**task, 'algorithm_timeout': .01}, cache, lambda m: None)
            self.assertEqual(record['status'], 'algo_timeout')
            graph = nx.empty_graph(3, create_using=nx.DiGraph)
            result = Result(Pattern(3), [graph], {})
            events = []
            with patch('mcpc.mcpc', return_value=result), patch(
                    'mcpc.metrics.evaluate', side_effect=lambda *args: time.sleep(.1)):
                record = runner._work({**task, 'algorithm': 'mcpc', 'params': {},
                                       'metric_timeout': .01}, cache, events.append)
            self.assertEqual(record['status'], 'metric_timeout')
            self.assertIsNotNone(record['pattern'])
            self.assertIn('runtime', record)
            self.assertTrue(any('partial' in event for event in events))
            with patch('mcpc.mcpc', return_value=result), patch(
                    'mcpc.metrics.evaluate', side_effect=runner.Cancelled):
                cancelled = runner._work({**task, 'algorithm': 'mcpc', 'params': {}}, cache, lambda m: None)
            self.assertEqual((cancelled['status'], cancelled['stage']), ('preempted', 'metric'))
            self.assertEqual(cancelled['pattern'], result.pattern.to_dict())

    def test_spawn_worker_resume_and_scientific_key(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results'
            task = tiny_task()
            records = runner.run([task], output, workers=1)
            self.assertEqual(records[0]['status'], 'ok')
            path = output / 'runs.jsonl'
            saved = path.read_bytes()
            resumed = runner.run([task], output, workers=1)
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(records[0]['key'], resumed[0]['key'])
            changed = {**task, 'params': {**task['params'], 'alpha': .2}}
            newer = runner.run([changed], output, workers=1)
            self.assertNotEqual(records[0]['key'], newer[0]['key'])
            self.assertEqual(len(runner.load_records(path)), 2)

    def test_spawn_worker_records_error_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            task = tiny_task()
            bad = {**task, 'id': 'bad', 'data_spec': {**task['data_spec'], 'kind': 'invalid'}}
            timeout = {**task, 'id': 'timeout', 'algorithm_timeout': .05,
                       'params': {**task['params'], 'pause': True}}
            with patch.object(runner, '_worker', sleeping_worker):
                records = runner.run([bad, timeout], Path(directory) / 'results', workers=1)
            statuses = {r['id']: r['status'] for r in records}
            self.assertEqual(statuses, {'bad': 'error', 'timeout': 'algo_timeout'})

    def test_timeout_recycles_worker_before_next_healthy_task(self):
        with tempfile.TemporaryDirectory() as directory:
            healthy = tiny_task('healthy')
            timeout = {**healthy, 'id': 'timeout', 'algorithm_timeout': .05,
                       'params': {**healthy['params'], 'pause': True}}
            with patch.object(runner, '_worker', sleeping_worker):
                records = runner.run([timeout, healthy], Path(directory) / 'results', workers=1)
            by_id = {record['id']: record for record in records}
            self.assertEqual(by_id['timeout']['status'], 'algo_timeout')
            self.assertEqual(by_id['healthy']['status'], 'ok')
            self.assertNotEqual(by_id['timeout']['worker_pid'], by_id['healthy']['worker_pid'])
            self.assertEqual(by_id['healthy']['metrics']['model_count'], 1)

    def test_data_timeout_is_retried_under_the_same_task_key(self):
        with tempfile.TemporaryDirectory() as directory:
            task = tiny_task()
            code = runner.code_digest()
            key = runner.digest({'task': task, 'code': code})
            path = runner.prepare_store(Path(directory) / 'results')
            runner.append_record(path, dict(task, key=key, code=code, status='data_timeout'))
            records = runner.run([task], path.parent, workers=1)
            self.assertEqual((records[0]['key'], records[0]['status']), (key, 'ok'))
            self.assertEqual(len(path.read_text().splitlines()), 2)


class InterruptionTests(unittest.TestCase):
    def test_sigterm_preserves_active_jobs_in_both_journals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ('random', 'pr')}
            tasks = []
            for name in paths:
                task = tiny_task(name)
                task['group']['experiment'] = name
                task['params'] = dict(wait_for=2, signal=signal.SIGTERM if name == 'random' else None)
                tasks.append(task)
            with patch.object(runner, '_worker', controlled_worker):
                with self.assertRaises(runner.RunInterrupted) as caught:
                    runner.run(tasks, root, workers=2, output_dirs=paths)
            self.assertEqual(len(caught.exception.records), 2)
            for name, path in paths.items():
                record = next(iter(runner.load_records(path / 'runs.jsonl').values()))
                self.assertEqual((record['group']['experiment'], record['status']), (name, 'preempted'))
                self.assertEqual(record['pattern'], Pattern(3).to_dict())
            self.assertFalse({r['worker_pid'] for r in caught.exception.records}
                             & {p.pid for p in mp.active_children()})

    def test_signals_preserve_buffered_completion_and_partial_pattern(self):
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for signum in old_handlers:
            for complete in (False, True):
                with self.subTest(signal=signum, complete=complete), tempfile.TemporaryDirectory() as directory:
                    task = tiny_task()
                    task['params'] = dict(signal=signum, complete=complete)
                    callback = []
                    with patch.object(runner, '_worker', controlled_worker):
                        with self.assertRaises(runner.RunInterrupted) as caught:
                            runner.run([task], Path(directory) / 'runs', workers=1, on_record=callback.append)
                    records = list(runner.load_records(Path(directory) / 'runs/runs.jsonl').values())
                    self.assertEqual(records, json.loads(json.dumps(caught.exception.records)))
                    self.assertEqual(caught.exception.signum, signum)
                    self.assertEqual(records[0]['status'], 'ok' if complete else 'preempted')
                    self.assertEqual(records[0]['pattern'], Pattern(3).to_dict())
                    self.assertEqual(records[0]['stage'], 'metric')
                    self.assertEqual(records[0]['runtime'], .125)
                    self.assertEqual(json.loads(json.dumps(callback)), records)
                    self.assertFalse(any(p.pid == records[0]['worker_pid'] for p in mp.active_children()))
        self.assertEqual({sig: signal.getsignal(sig) for sig in old_handlers}, old_handlers)

    def test_callback_failure_stops_both_workers_and_preserves_routed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ('random', 'pr')}
            tasks = []
            for name in paths:
                task = tiny_task(name)
                task['group']['experiment'] = name
                task['params'] = dict(complete=name == 'random', wait_for=2)
                tasks.append(task)
            def fail(record):
                saved = runner.load_records(paths[record['group']['experiment']] / 'runs.jsonl')
                self.assertIn(record['key'], saved)
                raise RuntimeError('plot callback failed')
            with patch.object(runner, '_worker', controlled_worker):
                with self.assertRaisesRegex(RuntimeError, 'plot callback failed'):
                    runner.run(tasks, root, workers=2, output_dirs=paths, on_record=fail)
            records = {name: next(iter(runner.load_records(path / 'runs.jsonl').values()))
                       for name, path in paths.items()}
            self.assertEqual(records['random']['status'], 'ok')
            self.assertEqual(records['pr']['status'], 'preempted')
            self.assertEqual(records['pr']['pattern'], Pattern(3).to_dict())
            self.assertFalse({r['worker_pid'] for r in records.values()} & {p.pid for p in mp.active_children()})

    def test_routing_loads_each_journal_once_and_callback_follows_fsync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ('random', 'pr')}
            tasks = []
            for name in paths:
                task = tiny_task(name)
                task['group']['experiment'] = name
                tasks.append(task)
            events, real_fsync = [], os.fsync
            def sync(fd):
                real_fsync(fd)
                events.append('fsync')
            with patch.object(runner, 'load_records', wraps=runner.load_records) as load, \
                    patch.object(runner.os, 'fsync', side_effect=sync):
                records = runner.run(tasks, root, workers=1, output_dirs=paths,
                                     on_record=lambda r: events.append('callback'))
            self.assertEqual(load.call_count, 2)
            self.assertEqual(events.count('callback'), 2)
            self.assertTrue(all(events[index-1] == 'fsync' for index, event in enumerate(events)
                                if event == 'callback'))
            self.assertTrue(all(r['status'] == 'ok' for r in records))
            self.assertEqual(len(list((root / 'models').glob('*.npz'))), 1)
            for name, path in paths.items():
                saved = list(runner.load_records(path / 'runs.jsonl').values())
                self.assertEqual([r['group']['experiment'] for r in saved], [name])


class ResumeSafetyTests(unittest.TestCase):
    def test_dependency_change_rejects_resume_without_relabeling_environment(self):
        environment = dict(code='revision', python='3.13', packages={'numpy': '2.3'})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, 'environment', return_value=environment):
                self.assertEqual(runner.run([], directory), [])
            path = Path(directory) / 'environment.json'
            original = path.read_bytes()
            changed = {**environment, 'packages': {'numpy': '2.4'}}
            with patch.object(runner, 'environment', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'environment changed'):
                    runner.run([], directory)
            self.assertEqual(path.read_bytes(), original)

    def test_chambers_resume_retries_failed_refinements_but_retains_timeouts(self):
        from experiments import __main__ as cli, chambers
        def record(task, status):
            return dict(task, code='revision', status=status,
                        key=runner.digest({'task': task, 'code': 'revision'}))
        calls, journal = [], {}
        def run(tasks, directory, workers, on_record=None):
            calls.append(tasks)
            outcomes = [record(task, 'ok') for task in tasks]
            journal.update({r['key']: r for r in outcomes})
            return outcomes
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / 'data.csv'
            csv.write_text('checksum-validated fixture; fitting is mocked')
            base = chambers._task('pc', .05, csv)
            extra = [chambers._task('pc', value, csv) for value in (.123, .234, .345, .456, .567)]
            saved = [record(base, 'ok'), *[record(task, status) for task, status in zip(
                extra, ('error', 'crashed', 'preempted', 'data_timeout', 'algo_timeout'))]]
            journal.update({r['key']: r for r in saved})
            with patch.object(chambers, 'tasks', return_value=iter([base])), patch.object(
                    chambers, 'next_points', return_value=[]), patch.object(cli, 'run', side_effect=run), \
                    patch.object(cli, 'load_records', return_value=journal), \
                    patch.object(cli, 'code_digest', return_value='revision'), \
                    patch('experiments.models.CHAMBERS_MD5', hashlib.md5(csv.read_bytes()).hexdigest()), \
                    patch.object(cli, 'render') as render, patch('sys.argv', ['experiments', 'chambers']):
                cli.main()
        self.assertEqual(len(calls), 2)
        self.assertEqual({task['id'] for task in calls[1]}, {task['id'] for task in extra[:4]})
        final = {r['id']: r['status'] for r in render.call_args.args[1]}
        self.assertEqual(final[extra[-1]['id']], 'algo_timeout')
        self.assertTrue(all(final[task['id']] == 'ok' for task in extra[:4]))

    def test_chambers_resume_rejects_tampered_csv_before_running(self):
        from experiments import __main__ as cli, chambers
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / 'data.csv'
            csv.write_text('modified dataset')
            output = root / 'results'
            store = runner.prepare_store(output / 'chambers')
            task, code = chambers._task('pc', .05, csv), runner.code_digest()
            runner.append_record(store, dict(task, code=code, status='ok',
                                            key=runner.digest({'task': task, 'code': code})))
            before = store.read_bytes()
            argv = ['experiments', 'chambers', '--out-dir', str(output), '--data-path', str(csv)]
            with patch('sys.argv', argv), patch.object(cli, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'expected official buchholz_1_obs.csv'):
                    cli.main()
            run.assert_not_called()
            self.assertEqual(store.read_bytes(), before)

    def test_chambers_plot_only_does_not_require_original_csv(self):
        from experiments import __main__ as cli, chambers
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / 'missing.csv'
            task = chambers._task('pc', .05, missing)
            code = runner.code_digest()
            record = dict(task, code=code, status='ok',
                          key=runner.digest({'task': task, 'code': code}))
            store = runner.prepare_store(root / 'chambers')
            runner.append_record(store, record)
            argv = ['experiments', 'chambers', '--plot-only', '--out-dir', str(root),
                    '--data-path', str(missing)]
            with patch('sys.argv', argv), patch.object(cli, 'render') as render, \
                    patch.object(cli, 'run') as run:
                cli.main()
            run.assert_not_called()
            self.assertEqual(render.call_args.args[1], [record])
            self.assertFalse(missing.exists())



if __name__ == '__main__':
    unittest.main()
