"""PR progress and serialized background figure updates."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

from .runner import digest

TASK_FIELDS = ('id', 'algorithm', 'data_spec', 'params', 'group',
               'algorithm_timeout', 'metric_timeout', 'total_timeout')
TERMINAL = {'ok', 'invalid', 'algo_timeout', 'metric_timeout', 'oom', 'oom_killed'}
STATUSES = ('ok', 'invalid', 'algo_timeout', 'metric_timeout', 'oom', 'oom_killed',
            'error', 'crashed', 'preempted', 'data_timeout')


def task_of(record):
    return {key: record[key] for key in TASK_FIELDS if key in record}


def identity(record):
    return digest(task_of(record))


class _Progress:
    def __init__(self, tasks):
        self.index, self.records = {}, {}
        self.expected, self.observed = Counter(), defaultdict(Counter)
        self.groups, self.counts = defaultdict(set), defaultdict(Counter)
        for task in tasks:
            key = identity(task)
            if key in self.index:
                continue
            spec, name = task['data_spec'], task['group']['experiment']
            row = name, spec['n_nodes'], task['algorithm']
            model = row, digest({k: v for k, v in spec.items() if k not in ('data_seed', 'n_samples')})
            self.index[key] = row, model
            self.expected[model] += 1
            self.groups[row].add(model)

    def store(self, record, strict=False):
        key = identity(record)
        if key not in self.index:
            if strict:
                raise ValueError(f"Unexpected task: {record['id']}")
            return
        status = record['status']
        if status not in STATUSES:
            raise ValueError(f'Unknown outcome: {status}')
        row, model = self.index[key]
        previous = self.records.get(key)
        if previous is not None:
            self.counts[row][previous['status']] -= 1
            self.observed[model][previous['status']] -= 1
        self.records[key] = record
        self.counts[row][status] += 1
        self.observed[model][status] += 1

    def rows(self):
        rows = []
        for key, models in sorted(self.groups.items()):
            counts = self.counts[key]
            rows.append(dict(zip(('experiment', 'n_nodes', 'algorithm'), key),
                target_models=len(models),
                started_models=sum(sum(self.observed[m].values()) > 0 for m in models),
                completed_models=sum(sum(self.observed[m][s] for s in TERMINAL) == self.expected[m]
                                     for m in models),
                fully_scored_models=sum(self.observed[m]['ok'] == self.expected[m] for m in models),
                expected_fits=sum(self.expected[m] for m in models),
                completed_fits=sum(counts[s] for s in TERMINAL),
                **{s: counts[s] for s in STATUSES}))
        return rows


def model_progress(tasks, records):
    progress = _Progress(tasks)
    for record in records.values() if isinstance(records, dict) else records:
        progress.store(record)
    return progress.rows()


def _atomic(path, text):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class Checkpoint(_Progress):
    def __init__(self, tasks, records, directory, plot_every=300):
        if plot_every <= 0:
            raise ValueError('plot_every must be positive')
        super().__init__(tasks)
        for record in records:
            self.store(record)
        self.root, self.plot_every = Path(directory), plot_every
        self.last_report = self.last_plot = time.monotonic()
        self.executor = self.pending_plot = None

    def _finish_plot(self, wait=False):
        if self.pending_plot is not None and (wait or self.pending_plot.done()):
            pending, self.pending_plot = self.pending_plot, None
            pending.result()
            self.last_plot = time.monotonic()

    def _render(self, records):
        from .plots import render
        if records:
            render('pr', records, self.root)

    def close(self):
        try:
            self._finish_plot(wait=True)
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.executor = None

    def add(self, record):
        self.store(record, strict=True)
        self._finish_plot()
        if record['status'] == 'error':
            self.snapshot('needs_attention', background=True)
            raise RuntimeError(f"{record['id']}: execution error; inspect saved runs.jsonl")
        now = time.monotonic()
        plots = self.pending_plot is None and now - self.last_plot >= self.plot_every
        if plots or now - self.last_report >= 30:
            self.snapshot(plots=plots, background=True)

    def snapshot(self, state='running', plots=False, *, background=False):
        try:
            return self._snapshot(state, plots, background)
        finally:
            if not background and state in ('complete', 'interrupted', 'needs_attention'):
                self.close()

    def _snapshot(self, state, plots, background):
        rows = self.rows()
        totals = {key: sum(row[key] for row in rows) for key in
                  ('target_models', 'started_models', 'completed_models', 'fully_scored_models',
                   'expected_fits', 'completed_fits')}
        if state == 'complete' and totals['completed_fits'] != totals['expected_fits']:
            raise ValueError('Cannot mark unfinished fits complete')
        payload = dict(state=state, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
                       totals=totals, rows=rows)
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic(self.root / 'progress.json', json.dumps(payload, indent=2) + '\n')
        self.last_report = time.monotonic()
        print(f"PR: {totals['completed_fits']}/{totals['expected_fits']} fits, "
              f"{totals['completed_models']}/{totals['target_models']} models; {state}", flush=True)
        batch = list(self.records.values()) if plots else []
        if background:
            self._finish_plot()
            if plots and self.pending_plot is None:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='figures')
                self.pending_plot = self.executor.submit(self._render, batch)
                self.last_plot = time.monotonic()
        else:
            if plots:
                self._finish_plot(wait=True)
                self._render(batch)
                self.last_plot = time.monotonic()
        return payload
