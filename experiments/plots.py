"""Publication figures and compact tables, generated only from saved records."""
import csv
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
import networkx as nx
import numpy as np

from mcpc.graph import Pattern, cpdag
from .summaries import operating_points, pr_auc

COLORS = dict(mcpc='#2563a6', pc='#bf423e', cpc='#39854a', ges='#8b59a0')
LABELS = dict(mcpc='mCPC', pc='PC', cpc='CPC', ges='GES')
METRICS = {'fd': 'FD', 'fdr': 'FDR', 'fn': 'FN', 'fnr': 'FNR',
           'shd': 'CPDAG SHD', 'rank': 'Adjacencies', 'n_amb': 'Ambiguous triples',
           'precision': 'Precision', 'recall': 'Recall', 'runtime': 'Algorithm time (s)'}
EDGE_STYLES = {'correct': ('#3a3936', 'solid', 1.6),
               'reversed': ('#d17b15', 'solid', 1.8),
               'extra': ('#d03b3b', 'dashed', 1.3)}


def setup():
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 18, 'legend.fontsize': 13.5,
                         'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'figure.dpi': 130,
                         'savefig.dpi': 180, 'pdf.fonttype': 42})


@contextmanager
def atomic_path(path):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.stem}-',
                                     suffix=path.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save(fig, directory, name, formats=('pdf', 'png')):
    try:
        for suffix in formats:
            with atomic_path(Path(directory) / f'{name}.{suffix}') as temporary:
                fig.savefig(temporary, bbox_inches='tight')
    finally:
        plt.close(fig)


def write_csv(path, rows, columns):
    with atomic_path(path) as temporary:
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, columns)
            writer.writeheader()
            writer.writerows(rows)


def write_table(directory, name, rows, note=''):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(Path(directory) / (name + '.csv'), rows, columns)
    def cell(value):
        if isinstance(value, float):
            return f'{value:.4g}'
        return str(value).replace('|', '/') if value is not None else '—'
    lines = [note, '', '| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(cell(row.get(k)) for k in columns) + ' |' for row in rows]
    with atomic_path(Path(directory) / (name + '.md')) as temporary:
        temporary.write_text('\n'.join(lines) + '\n')


def summary(records, directory):
    rows, grouped = [], defaultdict(list)
    for r in records:
        if r['algorithm'] == 'faithfulness':
            continue
        row = dict(id=r['id'], algorithm=r['algorithm'], status=r['status'],
                   parameters=json.dumps(r['params'], sort_keys=True), **r['group'])
        row.update({k: v for k, v in (r.get('metrics') or {}).items() if np.isscalar(v)})
        row.update(runtime=r.get('runtime'), metric_runtime=r.get('metric_runtime'))
        rows.append(row)
        key = tuple((k, r['group'][k]) for k in ('example', 'n_nodes', 'degree', 'n_samples', 'value')
                    if k in r['group']) + (('algorithm', r['algorithm']),)
        grouped[key].append(r)
    if rows:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        write_csv(Path(directory) / 'metrics.csv', rows, columns)
    aggregate = []
    for key, runs in grouped.items():
        good = [r for r in runs if r['status'] == 'ok']
        row = dict(key, contributing=len(good), attempted=len(runs))
        row['failures'] = json.dumps(dict(Counter(r['status'] for r in runs if r['status'] != 'ok')))
        for metric in METRICS:
            values = [r.get('runtime') if metric == 'runtime' else r['metrics'].get(metric)
                      for r in good]
            values = [v for v in values if v is not None]
            if values:
                row[metric] = float(np.mean(values))
                row[metric + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        aggregate.append(row)
    write_table(directory, 'summary', aggregate,
                'Means and sample SDs use fully scored outputs (status ok), including runtime. '
                'FD/FN and rates are upper bounds; precision/recall derive from those bounds. '
                'Attempted counts cover saved outcomes, including failures; pending jobs are excluded. '
                'The true-adjacency reference counts each available dataset once, regardless of fit status. '
                'For sweeps, value is α for PC/CPC/mCPC and the BIC penalty λ for GES.')


def truth_edge_reference(records):
    """Count each sampled dataset once, regardless of algorithm or fit outcome."""
    datasets = {json.dumps(r['data_spec'], sort_keys=True): r for r in records
                if r.get('truth') is not None}
    counts = defaultdict(list)
    for record in datasets.values():
        counts[record['group']['n_samples']].append(len(record['truth']))
    x = sorted(counts)
    y = [float(np.mean(counts[n])) for n in x]
    varying = len({value for values in counts.values() for value in values}) > 1
    return x, y, 'Mean true adjacencies' if varying else 'True adjacencies'


def comparison(records, directory, name):
    metrics = ('fd', 'fdr', 'rank', 'n_amb')
    columns = len(metrics) // 2
    fig, axes = plt.subplots(2, columns, figsize=(3.3*columns, 5.4), layout='constrained')
    for ax, metric in zip(axes.flat, metrics):
        for algorithm in ('mcpc', 'pc', 'cpc', 'ges'):
            buckets = defaultdict(list)
            for r in records:
                if r['algorithm'] == algorithm and r['status'] == 'ok':
                    value = r.get('runtime') if metric == 'runtime' else r['metrics'].get(metric)
                    if value is not None:
                        buckets[r['group']['n_samples']].append(value)
            if not buckets:
                continue
            x = sorted(buckets)
            mean = np.array([np.mean(buckets[n]) for n in x])
            sd = np.array([np.std(buckets[n]) for n in x])
            ax.plot(x, mean, 'o-', color=COLORS[algorithm], label=LABELS[algorithm], ms=3)
            ax.fill_between(x, np.maximum(mean-sd, 0), mean+sd, color=COLORS[algorithm], alpha=.12)
        if metric == 'rank':
            x, y, label = truth_edge_reference(records)
            if x:
                if len(set(y)) == 1:
                    line = ax.axhline(y[0], color='#222222', ls='--', lw=1.3,
                                      label=f'{label} = {y[0]:g}')
                else:
                    line, = ax.plot(x, y, '--', color='#222222', lw=1.3, label=label)
                legend = ax.legend(handles=[line], frameon=False, handlelength=1.2, handletextpad=.4)
                legend.get_texts()[0].set_text(line.get_label().replace(' adjacencies', '\nadjacencies'))
        ax.set(xscale='log', xlabel='Number of samples', ylabel=METRICS[metric])
    axes.flat[0].legend(frameon=False, ncols=2, loc='upper left', columnspacing=1, handlelength=1.5)
    save(fig, directory, name)


def faithfulness(records, directory):
    columns = ('faithfulness', 'adjacency_faithfulness', 'triangle_faithfulness', 'm1', 'm2', 'm3')
    labels = ('Full', 'Adjacency', 'Triangle', 'm = 1', 'm = 2', 'm = 3')
    by_n = defaultdict(list)
    for record in records:
        if record['status'] == 'ok':
            for row in record['metrics']['rows']:
                by_n[row['n_samples']].append(row)
    rows, fig = [], plt.figure(figsize=(6.2, 5), layout='constrained')
    ax = fig.subplots()
    for field, label in zip(columns, labels):
        x, y = [], []
        for n, results in sorted(by_n.items()):
            rate = float(np.mean([r[field] for r in results]))
            x.append(n)
            y.append(rate)
            rows.append(dict(condition=label, n_samples=n, violation_rate=rate,
                             contributing=len(results), attempted=len(records)))
        ax.plot(x, y, 'o-', label=label, ms=3)
    ax.set(xscale='log', xlabel='Number of samples', ylabel='Fraction of empirical violations', ylim=(-.02, 1.02))
    ax.legend(frameon=False, ncols=3, loc='lower center', bbox_to_anchor=(.5, 1.02), columnspacing=1)
    save(fig, directory, 'faithfulness')
    write_table(directory, 'violation_rates', rows)


def precision_recall(records, directory):
    points = operating_points(records)
    fig, ax = plt.subplots(figsize=(5.4, 5), layout='constrained')
    areas = []
    for algorithm in ('mcpc', 'pc', 'cpc', 'ges'):
        selected = [p for p in points if p['algorithm'] == algorithm and p['contributing']]
        if not selected:
            ax.plot([], [], color=COLORS[algorithm], label=f'{LABELS[algorithm]} (AUC unavailable)')
            areas.append(dict(algorithm=LABELS[algorithm], pr_auc=None, max_recall=None, points=0))
            continue
        coords = sorted({(0., 1.), *((p['recall'], p['precision']) for p in selected)})
        area = pr_auc(selected)
        ax.plot(*zip(*coords), 'o-', color=COLORS[algorithm],
                label=f'{LABELS[algorithm]} (AUC = {area:.3f})', ms=3)
        areas.append(dict(algorithm=LABELS[algorithm], pr_auc=area,
                          max_recall=max(p['recall'] for p in selected), points=len(selected)))
    ax.plot(0, 1, marker='o', mfc='white', mec='#333333', ms=6, zorder=5)
    ax.set(xlabel='Recall', ylabel='Precision', xlim=(-.02, 1.02), ylim=(-.02, 1.03))
    ax.legend(frameon=False, loc='lower center', ncols=2, columnspacing=.6,
              handlelength=1.2, handletextpad=.4, borderpad=.3)
    save(fig, directory, 'pr')
    write_table(directory, 'pr_auc', areas, 'Trapezoidal area; no extrapolation beyond measured recall.')
    write_table(directory, 'operating_points', points,
                'PR means use fully scored outputs. Attempted counts cover saved outcomes, '
                'including fit or metric failures; pending jobs are excluded.')


def scalability(records, directory):
    groups = defaultdict(list)
    for r in records:
        key = (r['group']['n_nodes'], r['group']['degree'])
        if r['algorithm'] == 'mcpc':
            groups[key].append(r)
    rows = []
    for key, runs in sorted(groups.items()):
        counts = Counter(r['status'] for r in runs)
        times = [r['runtime'] for r in runs if (r['stage'] == 'metric' or r['status'] in ('ok', 'invalid'))
                 and r.get('runtime') is not None]
        row = dict(zip(('n', 'degree'), key))
        row.update(median_seconds=float(np.median(times)) if times else None,
                   completed_algorithms=len(times), attempted=len(runs),
                   **{status: counts[status] for status in ('ok', 'invalid', 'algo_timeout',
                      'metric_timeout', 'data_timeout', 'preempted', 'error', 'crashed')})
        row.update({f'{stage}_{status}': sum(r['status'] == status and r['stage'] == stage
                                           for r in runs)
                    for stage in ('algorithm', 'metric', 'data') for status in ('oom', 'oom_killed')})
        rows.append(row)
    write_table(directory, 'scalability', rows,
                'mCPC: m=1, UGL penalty=.2, α=.2. '
                'Median algorithm time excludes metric evaluation and interrupted fits. '
                'Completed algorithms include invalid outputs and fits whose metric evaluation failed. '
                'Memory failures are split by algorithm, metric, and data stages; metric failures '
                'do not reduce the completed-algorithm count. '
                'Timeouts and memory failures are measured outcomes, not forecast exclusions.')


def draw_ambiguities(ax, positions, ambiguous):
    counts = Counter()
    for i, k, j in sorted(ambiguous, key=lambda t: (t[1], t[0], t[2])):
        cx, cy = positions[k]
        a, b = [np.degrees(np.arctan2(positions[v][1]-cy, positions[v][0]-cx))
                for v in (i, j)]
        start, span = a, (b-a) % 360
        if span > 180:
            start, span = b, 360-span
        radius = .4 * (1 + .16 * counts[k])
        counts[k] += 1
        ax.add_patch(Arc((cx, cy), 2*radius, 2*radius, theta1=start, theta2=start+span,
                         color='#2a78d6', lw=1.8, zorder=1.5))


def edge_kind(edge, directed, truth):
    if edge not in truth.directed and edge[::-1] not in truth.directed \
            and tuple(sorted(edge)) not in truth.undirected:
        return 'extra'
    return 'reversed' if directed and edge[::-1] in truth.directed else 'correct'


def graph_panel(ax, pattern, positions, labels, truth):
    ax.set_axis_off()
    ax.set_aspect('equal')
    graph = pattern.skeleton()
    draw_ambiguities(ax, positions, pattern.ambiguous)
    true_pairs = {frozenset(e) for e in truth.skeleton().edges}
    missing = true_pairs - {frozenset(e) for e in graph.edges}
    nx.draw_networkx_edges(graph, positions, edgelist=sorted(tuple(sorted(e)) for e in missing),
                           arrows=False, width=1.3, style='dotted', edge_color='#c3c2b7', ax=ax)
    for directed, edges in ((False, pattern.undirected), (True, pattern.directed)):
        for kind, (color, style, width) in EDGE_STYLES.items():
            selected = sorted(e for e in edges if edge_kind(e, directed, truth) == kind)
            layer = nx.DiGraph() if directed else nx.Graph()
            layer.add_nodes_from(graph)
            layer.add_edges_from(selected)
            arrows = dict(node_size=1000, arrowstyle='-|>', arrowsize=13) if directed else {}
            nx.draw_networkx_edges(layer, positions, arrows=directed, **arrows,
                                   width=width, style=style, edge_color=color, ax=ax)
    nx.draw_networkx_nodes(graph, positions, node_size=1000, node_color='#fcfcfb',
                           edgecolors='#52514e', linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(graph, positions, labels, font_size=17, ax=ax)
    ax.set(xlim=(-3.5, 3.5), ylim=(.19, 4.85))


def chamber_graphs(records, directory):
    graph_alpha = .01
    available = [r for r in records if r.get('pattern') and r['algorithm'] in LABELS]
    if not available:
        return
    reference = available[0]
    display = ('$R$', '$G$', '$B$', '$\\theta_1$', '$\\theta_2$', '$\\tilde C$',
               '$\\tilde\\theta_1$', '$\\tilde\\theta_2$', '$\\tilde I_1$', '$\\tilde I_2$', '$\\tilde I_3$')
    labels = dict(enumerate(display))
    coords = [(-1.5,2.4),(-.4,3.4),(.8,4.4),(-2.9,2.4),(2.2,3.4),
              (-1.8,.99),(-3,1.68),(3,1.68),(-.6,.64),(.6,.64),(1.8,.99)]
    positions = dict(enumerate(coords))
    truth = nx.DiGraph()
    truth.add_nodes_from(range(len(labels)))
    truth.add_edges_from(reference['truth'])
    selected_mcpc = next((r for r in available if r['algorithm'] == 'mcpc'
                         and r['params'].get('alpha') == graph_alpha
                         and r['params'].get('ugl_penalty') == .2
                         and r['params'].get('m') == 1), None)
    if selected_mcpc is not None:
        truth_pattern = cpdag(truth)
        ugl = Pattern(len(labels), set(),
                      {tuple(e) for e in selected_mcpc['diagnostics']['ugl_edges']}, set())
        overview = [(truth_pattern, 'True CPDAG'),
                    (ugl, 'mcPC - undirected graph'),
                    (Pattern.from_dict(selected_mcpc['pattern']),
                     rf'mCPC ($\alpha = {graph_alpha:g}$)')]
        for algorithm in ('pc', 'cpc', 'ges'):
            field, value, symbol = ('penalty', .5, r'\lambda') if algorithm == 'ges' \
                else ('alpha', graph_alpha, r'\alpha')
            record = next((r for r in records if r['algorithm'] == algorithm
                           and r['params'].get(field) == value
                           and r['data_spec'] == selected_mcpc['data_spec']), None)
            pattern = Pattern.from_dict(record['pattern']) if record and record.get('pattern') else None
            title = f'{LABELS[algorithm]} (${symbol} = {value:g}$)'
            if record is None or record['status'] != 'ok':
                title += '\n' + (record['status'] if record else 'not run')
            overview.append((pattern, title))
        fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.2), layout='constrained')
        for ax, (pattern, title) in zip(axes.flat, overview):
            if pattern is None:
                ax.set_axis_off()
                ax.text(.5, .5, 'No graph available', ha='center', va='center', fontsize=18, transform=ax.transAxes)
            else:
                graph_panel(ax, pattern, positions, labels, truth_pattern)
            ax.set_title(title, fontsize=21.6)
        legend = [Line2D([], [], color='#3a3936', label='True adjacency'),
                  Line2D([], [], color='#d17b15', marker='>', label='False orientation'),
                  Line2D([], [], color='#d03b3b', ls='--', label='Extra adjacency'),
                  Line2D([], [], color='#c3c2b7', ls=':', label='Missed adjacency'),
                  Line2D([], [], color='#2a78d6', label='Ambiguous triple')]
        fig.legend(handles=legend, loc='outside lower center', ncols=5, frameon=False, fontsize=18)
        save(fig, directory, 'cpdag_mcpc', formats=('png',))
    else:
        (Path(directory) / 'cpdag_mcpc.png').unlink(missing_ok=True)


def render(name, records, directory):
    setup()
    if name == 'pr':
        directory = Path(directory) / 'n10'
    render_group(name, records, directory)


def render_group(name, records, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_table(directory, 'status', [dict(status=status, count=count)
                                    for status, count in sorted(Counter(r['status'] for r in records).items())])
    if name == 'faithfulness':
        faithfulness(records, directory)
        return
    if name == 'scalability':
        scalability(records, directory)
        return
    summary(records, directory)
    if name == 'examples':
        for example in ('fig4', 'fig5', 'fig6'):
            comparison([r for r in records if r['group']['example'] == example], directory, example)
    elif name in ('pr', 'chambers'):
        precision_recall(records, directory)
        if name == 'chambers':
            chamber_graphs(records, directory)
