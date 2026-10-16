"""Build a CPU-only SAM2 followup report, including incomplete sweep directories.

    python -m experiments.sam2_profile.report_followup \
        --root runs/sam2_gpu_bottleneck/ab/20261016_followup \
        --output runs/sam2_gpu_bottleneck/ab/20261016_followup/report

Writes PREFIX.json and PREFIX.md (a supplied .json/.md suffix is also accepted).
Does not launch benchmarks, import torch, or change any source result artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

DEFAULT_ROOT = Path('runs/sam2_gpu_bottleneck/ab/20261016_followup')
RUN_NAMES = ('base1', 'candidate1', 'base2', 'candidate2')


def _read(path):
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as error:
        return None, f'{path}: {error}'
    if not isinstance(data, dict):
        return None, f'{path}: expected a JSON object'
    return data, None


def _compare(a, b):
    counts = {key: 0 for key in ('frame', 'ids', 'boxes', 'masks')}
    first = None
    maximum = 0.0
    for x, y in zip(a['rows'], b['rows']):
        for key in counts:
            if x[key] != y[key]:
                counts[key] += 1
                if first is None:
                    first = x['frame']
        if len(x['boxes']) == len(y['boxes']):
            for u, v in zip(x['boxes'], y['boxes']):
                maximum = max(maximum, max((abs(i - j) for i, j in zip(u, v)), default=0))
    return dict(identical=len(a['rows']) == len(b['rows']) and not any(counts.values()),
                differing_frames=counts, first_differing_frame=first, max_box_abs_delta=maximum)


def _derive_summary(runs):
    means = {kind: mean(runs[f'{kind}{repeat}']['wall_s'] for repeat in (1, 2))
             for kind in ('base', 'candidate')}
    return dict(mean_wall_s=means, speedup=means['base'] / means['candidate'],
                base_wall_s=[runs[f'base{repeat}']['wall_s'] for repeat in (1, 2)],
                candidate_wall_s=[runs[f'candidate{repeat}']['wall_s'] for repeat in (1, 2)],
                base_noise_percent=100 * abs(runs['base1']['wall_s'] - runs['base2']['wall_s']) / means['base'],
                base_repeat_identity=_compare(runs['base1'], runs['base2']),
                candidate_identity=[_compare(runs['base1'], runs[f'candidate{repeat}']) for repeat in (1, 2)])


def _known_all(values):
    if not values or any(not isinstance(value, bool) for value in values):
        return None
    return all(values)


def _number(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _directory(directory):
    errors = []
    summary, error = _read(directory / 'summary.json')
    if error:
        errors.append(error)
    runs = {}
    for name in RUN_NAMES:
        data, error = _read(directory / f'{name}.json')
        if error:
            errors.append(error)
        if data is not None:
            runs[name] = data
    source = 'summary.json' if summary is not None else None
    if summary is None and len(runs) == len(RUN_NAMES):
        try:
            summary = _derive_summary(runs)
            source = 'derived from four run JSON files'
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            errors.append(f'{directory}: incomplete run schema: {error}')
    summary = summary or {}
    means = summary.get('mean_wall_s', {})
    if not isinstance(means, dict):
        means = {}
    identities = summary.get('candidate_identity', [])
    if not isinstance(identities, list):
        identities = []
    candidate_exact = (_known_all([item.get('identical') if isinstance(item, dict) else None
                                  for item in identities]) if len(identities) == 2 else None)
    baseline_identity = summary.get('base_repeat_identity', {})
    if not isinstance(baseline_identity, dict):
        baseline_identity = {}
    base_exact = baseline_identity.get('identical')
    if not isinstance(base_exact, bool):
        base_exact = None
    complete = (all(_number(means.get(kind)) is not None for kind in ('base', 'candidate'))
                and _number(summary.get('speedup')) is not None
                and candidate_exact is not None and base_exact is not None)
    return dict(name=directory.name, directory=str(directory), status='complete' if complete else 'pending',
                summary_source=source, mean_base_wall_s=_number(means.get('base')),
                mean_candidate_wall_s=_number(means.get('candidate')),
                speedup=_number(summary.get('speedup')), candidate_exact_both_repeats=candidate_exact,
                baseline_repeat_exact=base_exact, baseline_noise_percent=_number(summary.get('base_noise_percent')),
                available_runs={name: {key: data.get(key) for key in ('wall_s', 'frames', 'case', 'patches')}
                                for name, data in runs.items()},
                missing_run_files=[name + '.json' for name in RUN_NAMES if name not in runs],
                summary=summary or None, errors=errors)


def _microtest(root, name):
    path = root / f'{name}.json'
    data, error = _read(path)
    result = dict(name=name, path=str(path), status='complete' if data is not None else 'pending',
                  data=data, error=error)
    if data is None:
        return result
    if name == 'cuda_graphs_micro_v2':
        rows = data.get('results', [])
        if isinstance(rows, list) and rows:
            flags = [row.get(key) for row in rows if isinstance(row, dict)
                     for key in ('equal', 'dynamic_inputs_equal', 'retained_output_equal')]
            flags += [data.get('bounded_cache_fallback_equal'), data.get('autocast_boundary_equal')]
            speeds = [_number(row.get('wall_speedup')) for row in rows if isinstance(row, dict)]
            result['measured'] = dict(shapes=len(rows), exact_all_reported_checks=_known_all(flags),
                                      wall_speedups=[speed for speed in speeds if speed is not None])
    elif name == 'blas_micro':
        rows = data.get('rows', [])
        if isinstance(rows, list) and rows:
            speeds = []
            for row in rows:
                timings = row.get('ms', {}) if isinstance(row, dict) else {}
                base, candidate = _number(timings.get('cublas')), _number(timings.get('cublaslt'))
                if base is not None and candidate is not None and candidate > 0:
                    speeds.append(base / candidate)
            result['measured'] = dict(shapes=len(rows),
                                      exact_all_reported_shapes=_known_all([row.get('identical') for row in rows]),
                                      cublas_over_cublaslt_speedups=speeds)
    return result


def collect(root):
    candidate_root, stack_root = root / 'candidates', root / 'stack'
    candidates = sorted(path for path in candidate_root.iterdir() if path.is_dir()) if candidate_root.exists() else []
    stack_names = {'han', 'felix', 'bhc'}
    if stack_root.exists():
        stack_names.update(path.name for path in stack_root.iterdir() if path.is_dir())
    return dict(root=str(root), candidates=[_directory(path) for path in candidates],
                stack=[_directory(stack_root / name) for name in sorted(stack_names)],
                microtests=[_microtest(root, name) for name in ('cuda_graphs_micro_v2', 'blas_micro')])


def _format(value, digits=2, suffix=''):
    return 'pending' if value is None else f'{value:.{digits}f}{suffix}'


def _exact(value):
    return 'pending' if value is None else ('yes' if value else 'no')


def _table(rows):
    lines = ['| Configuration | Base s | Candidate s | Speedup | Exact 2/2 | Base noise | Base repeat exact | Status |',
             '|---|---:|---:|---:|---|---:|---|---|']
    for row in rows:
        name = row['name'].replace('|', '\\|')
        lines.append('| ' + ' | '.join((name, _format(row['mean_base_wall_s']),
                                       _format(row['mean_candidate_wall_s']), _format(row['speedup'], 3, 'x'),
                                       _exact(row['candidate_exact_both_repeats']),
                                       _format(row['baseline_noise_percent'], 2, '%'),
                                       _exact(row['baseline_repeat_exact']), row['status'])) + ' |')
    return '\n'.join(lines) if rows else 'Pending: no configuration directories exist yet.'


def markdown(report):
    lines = ['# SAM2 GB10 followup results', '',
             'Candidate trials use fused RoPE as baseline; final stack trials use the unpatched baseline. '
             'Times are mean complete-loop wall seconds over two alternating repeats. '
             'Exact 2/2 compares both candidate runs with the first baseline. '
             'Baseline noise is the absolute difference between baseline walls divided by their mean.',
             '', '## Candidates', '', _table(report['candidates']), '', '## Final stack by clip', '',
             _table(report['stack']), '', '## Microtest evidence', '']
    for micro in report['microtests']:
        measured = micro.get('measured')
        if measured is None:
            lines.append(f'- {micro["name"]}: pending JSON evidence.')
            continue
        if micro['name'] == 'cuda_graphs_micro_v2':
            speeds = measured['wall_speedups']
            interval = (f'{min(speeds):.3f}–{max(speeds):.3f}x wall speedup' if speeds else 'timing pending')
            lines.append(f'- CUDA graphs v2: {measured["shapes"]} measured shapes; '
                         f'all reported exactness checks {_exact(measured["exact_all_reported_checks"])}; {interval}.')
        else:
            speeds = measured['cublas_over_cublaslt_speedups']
            interval = (f'{min(speeds):.3f}–{max(speeds):.3f}x per-shape cuBLAS/cuBLASLt speedup' if speeds else 'timing pending')
            lines.append(f'- BLAS selection: {measured["shapes"]} measured shapes; '
                         f'all shapes exact {_exact(measured["exact_all_reported_shapes"])}; {interval}.')
    lines += ['', 'Microtest ratios describe isolated operations; the complete-loop tables determine end-to-end gains.']
    pending = [row for row in report['candidates'] + report['stack'] if row['status'] != 'complete']
    if pending:
        lines += ['', '## Pending evidence', '']
        for row in pending:
            missing = ', '.join(row['missing_run_files']) or 'complete summary fields'
            lines.append(f'- {row["name"]}: awaiting {missing}.')
    errors = [error for row in report['candidates'] + report['stack'] for error in row['errors']]
    errors += [micro['error'] for micro in report['microtests'] if micro.get('error')]
    if errors:
        lines += ['', 'Read errors (a running sweep may still be writing these files):', '']
        lines.extend('- ' + error for error in errors)
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, required=True, help='output prefix, or a .json/.md report path')
    args = parser.parse_args()
    report = collect(args.root)
    prefix = args.output.with_suffix('') if args.output.suffix in ('.json', '.md') else args.output
    json_path, markdown_path = Path(str(prefix) + '.json'), Path(str(prefix) + '.md')
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    markdown_path.write_text(markdown(report))
    print(json.dumps(dict(json=str(json_path), markdown=str(markdown_path),
                         complete_candidates=sum(row['status'] == 'complete' for row in report['candidates']),
                         complete_stack_clips=sum(row['status'] == 'complete' for row in report['stack']))))


if __name__ == '__main__':
    main()
