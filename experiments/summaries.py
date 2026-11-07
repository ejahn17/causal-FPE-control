"""Precision-recall summaries of saved experiment outcomes."""
import numpy as np


def operating_points(records):
    """Mean PR by algorithm and setting, including unsuccessful attempts in counts."""
    grouped = {}
    for record in records:
        if record['algorithm'] not in ('mcpc', 'pc', 'cpc', 'ges'):
            continue
        value = record.get('group', {}).get('value')
        if value is None:
            value = record['params'].get('alpha', record['params'].get('penalty'))
        grouped.setdefault((record['algorithm'], value), []).append(record)
    rows = []
    for (algorithm, value), runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        good = [r['metrics'] for r in runs if r['status'] == 'ok']
        rows.append(dict(algorithm=algorithm, value=value,
            precision=float(np.mean([m['precision'] for m in good])) if good else None,
            recall=float(np.mean([m['recall'] for m in good])) if good else None,
            contributing=len(good), attempted=len(runs), failed=len(runs)-len(good)))
    return rows


def pr_auc(points):
    coordinates = sorted({(0., 1.), *((p['recall'], p['precision']) for p in points
                                     if p['recall'] is not None and p['precision'] is not None)})
    return float(np.trapezoid([p for r, p in coordinates], [r for r, p in coordinates]))
