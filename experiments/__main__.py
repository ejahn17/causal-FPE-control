"""Run, resume, or redraw the publication experiments."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import hashlib
import importlib
import json
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / '.cache' / 'matplotlib'))

from .plots import render
from .runner import (RETRY, RunInterrupted, code_digest, digest, environment,
                     load_records, run as execute)

NAMES = ('faithfulness', 'examples', 'pr', 'scalability', 'chambers')
ALGORITHMS = ('mcpc', 'pc', 'cpc', 'ges', 'faithfulness')
TERMINAL = {'ok', 'invalid', 'algo_timeout', 'metric_timeout', 'oom', 'oom_killed'}
TASK_FIELDS = ('id', 'algorithm', 'data_spec', 'params', 'group',
               'algorithm_timeout', 'metric_timeout', 'total_timeout')


def task_of(record):
    return {key: record[key] for key in TASK_FIELDS if key in record}


def compatible_records(saved, current=None):
    """Verified reuse preserves each record's original execution code and key."""
    current = current or code_digest()
    selected = {}
    for record in saved:
        if record['algorithm'] not in ALGORITHMS:
            continue
        if record['code'] != current and record.get('validated_for') != current:
            continue
        task = task_of(record)
        if record['key'] != digest({'task': task, 'code': record['code']}):
            continue
        identity = digest(task)
        before = selected.get(identity)
        if before is None or record['code'] == current or before['code'] != current:
            selected[identity] = record
    return list(selected.values())


def cache_plan(tasks, saved, current=None):
    indexed = {digest(task_of(r)): r for r in compatible_records(saved, current)}
    cached, missing = [], []
    for task in tasks:
        record = indexed.get(digest(task))
        if record is None or record['status'] not in TERMINAL:
            missing.append(task)
        else:
            cached.append(record)
    return cached, missing


def check_environment(directory):
    path = Path(directory) / 'environment.json'
    if path.exists():
        before, now = json.loads(path.read_text()), environment()
        if {k: v for k, v in before.items() if k != 'code'} != {
                k: v for k, v in now.items() if k != 'code'}:
            raise ValueError('Execution environment changed; use a new --out-dir')
    elif (Path(directory) / 'runs.jsonl').exists():
        raise ValueError(f'Missing execution environment in {directory}')


def run(tasks, directory, workers=4, on_record=None):
    check_environment(directory)
    saved = load_records(Path(directory) / 'runs.jsonl')
    cached, missing = cache_plan(tasks, saved.values())
    return cached + (execute(missing, directory, workers, on_record=on_record) if missing else [])


def select_records(saved, tasks, name):
    records = compatible_records(saved)
    if name == 'chambers':
        def family(task):
            fixed = {k: v for k, v in task['params'].items() if k not in ('alpha', 'penalty')}
            return digest([task['algorithm'], task['data_spec'], fixed])
        families = {family(task) for task in tasks}
        return [r for r in records if family(r) in families]
    keys = {digest(task) for task in tasks}
    return [r for r in records if digest(task_of(r)) in keys]


def run_pr(tasks, records, directory, args):
    from .progress import Checkpoint
    check_environment(directory)
    checkpoint = Checkpoint(tasks, records, directory, args.plot_every)
    def refresh():
        # Worker cleanup can save outcomes after a completion callback fails.
        journal = load_records(directory / 'runs.jsonl')
        for record in select_records(journal.values(), tasks, 'pr'):
            checkpoint.store(record)
    try:
        checkpoint.snapshot(plots=True)
        for attempt in range(3):
            _, pending = cache_plan(tasks, checkpoint.records.values())
            if not pending:
                break
            print(f'PR: {len(pending)} fits remaining (attempt {attempt+1})', flush=True)
            for record in execute(pending, directory, args.workers, on_record=checkpoint.add):
                checkpoint.store(record)
        if cache_plan(tasks, checkpoint.records.values())[1]:
            raise RuntimeError('Unresolved interruptions after three attempts; resume to retry')
        checkpoint.snapshot('complete', plots=True)
    except (RunInterrupted, KeyboardInterrupt):
        refresh()
        checkpoint.snapshot('interrupted', plots=True)
        raise
    except Exception:
        refresh()
        checkpoint.snapshot('needs_attention', plots=True)
        raise
    finally:
        checkpoint.close()


def regular(name, args):
    driver = importlib.import_module('experiments.' + name)
    tasks = list(driver.tasks(data_path=args.data_path))
    directory = args.out_dir / name
    saved = load_records(directory / 'runs.jsonl')
    records = select_records(saved.values(), tasks, name)
    if args.plot_only:
        if not records:
            raise ValueError(f'No compatible saved outcomes in {directory}')
        render(name, records, directory)
        return
    if name == 'pr':
        run_pr(tasks, records, directory, args)
        return
    if name == 'chambers':
        from .models import CHAMBERS_MD5
        path = tasks[0]['data_spec']['path']
        if hashlib.md5(Path(path).read_bytes()).hexdigest() != CHAMBERS_MD5:
            raise ValueError(f'{path}: expected official buchholz_1_obs.csv ({CHAMBERS_MD5})')
    indexed = {digest(task_of(r)): r for r in records}
    def completed(record):
        indexed[digest(task_of(record))] = record
        if record['status'] == 'error':
            raise RuntimeError(f"{record['id']}: execution error; inspect {directory/'runs.jsonl'}")
    try:
        for record in run(tasks, directory, args.workers, completed):
            indexed[digest(task_of(record))] = record
        if name == 'chambers':
            retries = [task_of(r) for r in indexed.values() if r['status'] in RETRY]
            if retries:
                for record in run(retries, directory, args.workers, completed):
                    indexed[digest(task_of(record))] = record
            while proposals := driver.next_points(list(indexed.values())):
                for record in run(proposals, directory, args.workers, completed):
                    indexed[digest(task_of(record))] = record
    finally:
        journal = load_records(directory / 'runs.jsonl')
        for record in select_records(journal.values(), tasks, name):
            indexed[digest(task_of(record))] = record
        if indexed:
            render(name, list(indexed.values()), directory)
    if any(r['status'] not in TERMINAL for r in indexed.values()):
        raise RuntimeError(f'{name}: unresolved execution failures; resume to retry')
    print(f'{name}: {len(indexed)} saved outcomes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=(*NAMES, 'all'))
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out-dir', type=Path, default=Path('results'))
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--plot-every', type=float, default=300, help='seconds between figure updates')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.plot_every <= 0:
        parser.error('Require 1 <= workers <= 4 and plot-every > 0')
    try:
        for name in NAMES if args.experiment == 'all' else (args.experiment,):
            regular(name, args)
    except (RunInterrupted, KeyboardInterrupt):
        print('Interrupted; completed jobs are saved. Run the same command to resume.', flush=True)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
