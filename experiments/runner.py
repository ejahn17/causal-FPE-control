"""Resumable records and bounded persistent workers."""
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import tempfile
import time
import traceback

import psutil

RETRY = {'preempted', 'crashed', 'error', 'data_timeout'}
GB = 1024**3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def code_digest():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / 'mcpc').glob('*.py')) + sorted(
        p for p in (root / 'experiments').glob('*.py')
        if p.name not in ('plots.py', '__main__.py', '__init__.py'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def environment():
    names = ['numpy', 'scipy', 'networkx', 'scikit-learn', 'causal-learn', 'matplotlib', 'psutil']
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), cpu_count=os.cpu_count(), code=code_digest(),
                packages={name: importlib.metadata.version(name) for name in names})


def load_records(path):
    records = {}
    if Path(path).exists():
        lines = Path(path).read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break  # A process interruption can leave one incomplete final append.
                raise
            records[record['key']] = record
    return records


def append_record(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare_store(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'runs.jsonl'
    # Remove only a torn final write before appending new valid records.
    if path.exists():
        raw = path.read_bytes()
        if raw:
            stripped = raw.rstrip(b'\n')
            last = stripped.rfind(b'\n') + 1
            try:
                json.loads(stripped[last:])
                if not raw.endswith(b'\n'):
                    with path.open('ab') as stream:
                        stream.write(b'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
            except json.JSONDecodeError:
                with path.open('r+b') as stream:
                    stream.truncate(last)
                    stream.flush()
                    os.fsync(stream.fileno())
    return path


class TimeLimit(Exception):
    pass


class Cancelled(Exception):
    pass


class RunInterrupted(KeyboardInterrupt):
    """The pool stopped; records contains all durable outcomes for the requested tasks."""
    def __init__(self, records, signum):
        super().__init__(f'{signal.Signals(signum).name}: progress saved')
        self.records, self.signum = records, signum


@contextlib.contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeLimit()
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _work(task, cache, emit):
    from mcpc import mcpc
    from mcpc.baselines import fit, InvalidGraph
    from mcpc.metrics import evaluate
    from .models import materialize
    record = dict(task, status='error', metrics=None, pattern=None, worker_pid=os.getpid(),
                  started_at=datetime.now(timezone.utc).isoformat())
    stage, started, total_started = 'data', time.monotonic(), time.monotonic()
    try:
        emit({'stage': stage, 'limit': 300})
        with deadline(300):
            X, truth, metadata = materialize(task['data_spec'], cache)
        record.update(metadata=metadata, truth=sorted(map(list, truth.edges)),
                      n_nodes=len(truth), n_samples=len(X))
        emit({'partial': record})
        stage, started = 'algorithm', time.monotonic()
        emit({'stage': stage, 'limit': task['algorithm_timeout']})
        with deadline(task['algorithm_timeout']):
            if task['algorithm'] == 'faithfulness':
                from .faithfulness import evaluate as faithfulness
                record['metrics'] = faithfulness(X, truth, task['params'])
                result = None
            else:
                if task['algorithm'] == 'mcpc':
                    result = mcpc(X, **task['params'])
                elif task['algorithm'] in ('pc', 'cpc', 'ges'):
                    result = fit(X, task['algorithm'], task['params'])
                else:
                    raise ValueError(f"Unknown algorithm: {task['algorithm']}")
        record['runtime'] = time.monotonic() - started
        if result is not None:
            record.update(pattern=result.pattern.to_dict(), diagnostics=result.diagnostics)
            emit({'partial': record})
            stage, started = 'metric', time.monotonic()
            limit = task['metric_timeout']
            if task.get('total_timeout'):
                limit = min(limit, max(.001, task['total_timeout'] - record['runtime']))
            emit({'stage': stage, 'limit': limit})
            with deadline(limit):
                record['metrics'] = evaluate(result.pattern, result.representatives, truth)
            record['metric_runtime'] = time.monotonic() - started
        record['status'] = 'ok'
    except Cancelled:
        record['status'] = 'preempted'
    except InvalidGraph as exc:
        record.update(status='invalid', error=str(exc), metrics=getattr(exc, 'metrics', None))
        if getattr(exc, 'pattern', None) is not None:
            record['pattern'] = exc.pattern.to_dict()
    except TimeLimit:
        record['status'] = {'algorithm': 'algo_timeout', 'metric': 'metric_timeout',
                            'data': 'data_timeout'}[stage]
    except MemoryError:
        record['status'] = 'oom'
    except Exception:
        record.update(status='error', error=traceback.format_exc())
    record.update(stage=stage, elapsed=time.monotonic() - total_started)
    if stage == 'algorithm' and 'runtime' not in record:
        record['runtime'] = time.monotonic() - started
    if stage == 'metric' and 'metric_runtime' not in record:
        record['metric_runtime'] = time.monotonic() - started
    return record


def _worker(connection, cache):
    # Terminal interrupts belong to the parent, which drains and stops the pool.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    from mcpc.limits import set_memory_budget
    from threadpoolctl import threadpool_limits
    import causallearn.search.ConstraintBased.PC
    import causallearn.search.ScoreBased.GES
    import causallearn.utils.DAG2CPDAG
    threadpool_limits(limits=1)
    set_memory_budget(3)
    computing, stopping = False, False
    def memory_signal(*_):
        raise MemoryError('watchdog memory limit')
    def cancel(*_):
        nonlocal stopping
        stopping = True
        if computing:
            raise Cancelled()
    signal.signal(signal.SIGUSR1, memory_signal)
    signal.signal(signal.SIGTERM, cancel)
    try:
        while True:
            task = connection.recv()
            if task is None:
                return
            computing = True
            try:
                record = _work(task, cache, connection.send)
            finally:
                computing = False
            connection.send({'record': record})
            if stopping or record['status'] == 'preempted':
                return
    except (EOFError, BrokenPipeError, Cancelled):
        return


def run(tasks, directory, workers=4, on_record=None, output_dirs=None):
    """Run missing jobs; optionally route suites to output_dirs, sharing directory/models.

    Call on_record after fsync; RunInterrupted carries all durable requested outcomes.
    """
    if not 1 <= workers <= 4:
        raise ValueError('workers must be 1..4 (4 x 4 GB hard limit)')
    directory = Path(directory)
    directories = {name: Path(path) for name, path in output_dirs.items()} if output_dirs else {None: directory}
    tasks = list(tasks)
    if output_dirs and any(task['group']['experiment'] not in directories for task in tasks):
        raise ValueError('Every task experiment needs an output directory')
    env = environment()
    for path in directories.values():
        environment_path = path / 'environment.json'
        if environment_path.exists():
            previous_environment = json.loads(environment_path.read_text())
            if {k: v for k, v in previous_environment.items() if k != 'code'} != {
                    k: v for k, v in env.items() if k != 'code'}:
                raise ValueError('Execution environment changed; use a new --out-dir')
    paths = {name: prepare_store(path) for name, path in directories.items()}
    for path in directories.values():
        with tempfile.NamedTemporaryFile(mode='w', dir=path, prefix='.environment-', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(json.dumps(env, indent=2) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, path / 'environment.json')
            finally:
                temporary.unlink(missing_ok=True)
    cache = (directory if output_dirs else directory.parent) / 'models'
    cache.mkdir(parents=True, exist_ok=True)
    tasks = [dict(task, key=digest({'task': task, 'code': env['code']}), code=env['code'])
             for task in tasks]
    previous = {key: record for path in paths.values() for key, record in load_records(path).items()}
    pending = [task for task in tasks if task['key'] not in previous or
               previous[task['key']]['status'] in RETRY]
    desired = {task['key'] for task in tasks}
    if not pending:
        return [record for key, record in previous.items() if key in desired]
    ctx = mp.get_context('spawn')
    slots, completed, cursor = [], 0, 0
    interrupted, callback_error = None, None

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = interrupted or signum

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}

    def launch():
        parent, child = ctx.Pipe()
        process = ctx.Process(target=_worker, args=(child, str(cache)))
        try:
            process.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        return dict(process=process, pipe=parent, task=None, partial={}, stage='data',
                    started=0, limit=300, signal_at=None)

    def save(slot, status=None, record=None):
        nonlocal completed, callback_error
        if record is None:
            record = dict(slot['task'], **{k: v for k, v in slot['partial'].items()
                                         if k not in slot['task']})
            record.update(status=status, stage=slot['stage'], worker_pid=slot['process'].pid,
                          elapsed=time.monotonic()-slot['job_start'])
            if slot['stage'] == 'algorithm' and record.get('runtime') is None:
                record['runtime'] = time.monotonic() - slot['started']
            if slot['stage'] == 'metric':
                record['metric_runtime'] = time.monotonic() - slot['started']
        path = paths[record['group']['experiment'] if output_dirs else None]
        append_record(path, record)
        previous[record['key']] = record
        slot['task'] = None
        completed += 1
        if completed == 1 or completed % 25 == 0 or record['status'] == 'error':
            print(f'{directory.name}: {completed}/{len(pending)} new jobs; '
                  f'{record["id"]}: {record["status"]}', flush=True)
            if record['status'] == 'error':
                print(record.get('error', ''), flush=True)
        if on_record is not None and callback_error is None:
            try:
                on_record(record)
            except BaseException as exc:
                callback_error = exc

    def drain(slot):
        recycle = False
        while not slot['pipe'].closed and slot['pipe'].poll():
            try:
                message = slot['pipe'].recv()
            except (EOFError, OSError):
                break
            if 'record' in message:
                record = message['record']
                if slot['task'] is None or record['key'] != slot['task']['key']:
                    raise RuntimeError('Worker returned an unexpected task')
                save(slot, record=record)
                recycle = record['status'] in ('oom', 'algo_timeout', 'metric_timeout', 'data_timeout')
            elif 'partial' in message:
                slot['partial'] = message['partial']
            else:
                slot.update(stage=message['stage'], limit=message['limit'], started=time.monotonic())
        return recycle

    def stop(slot, status):
        if slot['process'].is_alive():
            slot['process'].kill()
        slot['process'].join()
        drain(slot)  # A completed result can already be buffered when a watchdog fires.
        if slot['task'] is not None:
            save(slot, status)
        slot['pipe'].close()

    def shutdown():
        error = None
        for slot in slots:
            try:
                drain(slot)
                if slot['process'].is_alive():
                    if slot['task'] is None:
                        slot['pipe'].send(None)
                    else:
                        slot['process'].terminate()  # SIGTERM returns a partial preempted record.
            except BaseException as exc:
                error = error or exc
        until = time.monotonic() + 1
        while time.monotonic() < until and any(s['process'].is_alive() for s in slots):
            for slot in slots:
                try:
                    drain(slot)
                except BaseException as exc:
                    error = error or exc
            time.sleep(.02)
        for slot in slots:
            try:
                stop(slot, 'preempted')
            except BaseException as exc:
                error = error or exc
            finally:
                if slot['process'].is_alive():
                    slot['process'].kill()
                slot['process'].join()
                slot['pipe'].close()
        return error

    failure = None
    try:
        for _ in range(min(workers, len(pending))):
            if interrupted:
                break
            slots.append(launch())
        while not interrupted and callback_error is None and (
                cursor < len(pending) or any(slot['task'] is not None for slot in slots)):
            for index, slot in enumerate(slots):
                recycle = drain(slot)
                if interrupted or callback_error is not None:
                    break
                process = slot['process']
                if recycle or not process.is_alive():
                    stop(slot, 'crashed')
                    if interrupted or callback_error is not None:
                        break
                    slot = slots[index] = launch()
                if interrupted or callback_error is not None:
                    break
                if slot['task'] is None and cursor < len(pending):
                    task = pending[cursor]
                    cursor += 1
                    slot.update(task=task, partial={}, started=time.monotonic(),
                                job_start=time.monotonic(), limit=300, stage='data', signal_at=None)
                    slot['pipe'].send(task)
            if interrupted or callback_error is not None:
                break
            rss = []
            for slot in slots:
                try:
                    memory = psutil.Process(slot['process'].pid).memory_info().rss
                except psutil.NoSuchProcess:
                    memory = 0
                rss.append(memory)
            for index, (slot, memory) in enumerate(zip(slots, rss)):
                if slot['task'] is None:
                    continue
                now = time.monotonic()
                status = None
                if memory > 4*GB or (slot['signal_at'] and now-slot['signal_at'] > 5):
                    status = 'oom_killed'
                elif memory > 3.5*GB and slot['signal_at'] is None:
                    try:
                        os.kill(slot['process'].pid, signal.SIGUSR1)
                        slot['signal_at'] = now
                    except ProcessLookupError:
                        pass
                elif now - slot['started'] > slot['limit'] + 5:
                    status = {'algorithm': 'algo_timeout', 'metric': 'metric_timeout',
                              'data': 'data_timeout'}[slot['stage']]
                if status:
                    stop(slot, status)
                    if not interrupted and callback_error is None:
                        slots[index] = launch()
            if sum(rss) > 12*GB or psutil.virtual_memory().available < 4*GB:
                active = [(memory, index) for index, (slot, memory) in enumerate(zip(slots, rss))
                          if slot['task'] is not None and memory > 256*1024**2]
                if active:
                    _, index = max(active)
                    stop(slots[index], 'preempted')
                    if not interrupted and callback_error is None:
                        slots[index] = launch()
            time.sleep(.1)
    except BaseException as exc:
        failure = exc
    finally:
        cleanup_error = shutdown()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    records = [record for key, record in previous.items() if key in desired]
    if failure or callback_error or cleanup_error:
        raise failure or callback_error or cleanup_error
    if interrupted:
        raise RunInterrupted(records, interrupted)
    return records
