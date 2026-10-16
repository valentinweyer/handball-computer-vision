"""Serialize the requested alternating full-clip A/B and final stack trials.

All subprocesses share one GPU and are deliberately run one at a time. Existing
JSON/log paths are never overwritten. Each record keeps process elapsed time
(including imports, warmup and setup) separately from the harness loop wall.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROPE = 'experiments.sam2_profile.fused_rope:install'
ROOT = Path(__file__).resolve().parents[2]


def compare(a, b):
    counts = {k: 0 for k in ('frame', 'ids', 'boxes', 'masks')}
    first = None
    max_box_delta = 0.0
    for x, y in zip(a['rows'], b['rows']):
        for k in counts:
            if x[k] != y[k]:
                counts[k] += 1
                if first is None:
                    first = x['frame']
        if len(x['boxes']) == len(y['boxes']):
            for u, v in zip(x['boxes'], y['boxes']):
                max_box_delta = max(max_box_delta, max((abs(i-j) for i,j in zip(u,v)), default=0))
    return {'identical':len(a['rows']) == len(b['rows']) and not any(counts.values()),
            'differing_frames':counts, 'first_differing_frame':first, 'max_box_abs_delta':max_box_delta}


def summarize(directory):
    values = {f'{kind}{repeat}': json.loads((directory/f'{kind}{repeat}.json').read_text())
              for repeat in (1,2) for kind in ('base','candidate')}
    means = {kind: sum(values[f'{kind}{r}']['wall_s'] for r in (1,2))/2
             for kind in ('base','candidate')}
    result = {'mean_wall_s':means, 'speedup':means['base']/means['candidate'],
              'base_wall_s':[values[f'base{r}']['wall_s'] for r in (1,2)],
              'candidate_wall_s':[values[f'candidate{r}']['wall_s'] for r in (1,2)],
              'base_noise_percent':100*abs(values['base1']['wall_s']-values['base2']['wall_s'])/means['base'],
              'base_repeat_identity':compare(values['base1'],values['base2']),
              'candidate_identity':[compare(values['base1'],values[f'candidate{r}']) for r in (1,2)]}
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def run_one(case, specs, path):
    if path.exists() or path.with_suffix('.log').exists():
        raise FileExistsError(f'Refusing to overwrite existing benchmark: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable,'-m','experiments.sam2_profile.ab','run','--case',case,'--output',str(path)]
    for spec in specs:
        command.extend(['--patch',spec])
    hashes = {}
    for spec in specs:
        source = ROOT/Path(spec.split(':')[0].replace('.','/')).with_suffix('.py')
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    print(json.dumps({'event':'start','case':case,'path':str(path),'patches':specs}),flush=True)
    started = time.monotonic()
    with path.with_suffix('.log').open('x') as log:
        completed = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    record = {'command':command,'returncode':completed.returncode,
              'process_elapsed_s':time.monotonic()-started,'patch_sha256':hashes}
    path.with_suffix('.process.json').write_text(json.dumps(record,indent=2)+'\n')
    if completed.returncode:
        raise RuntimeError(f'Benchmark failed; see {path.with_suffix(".log")}')
    result = json.loads(path.read_text())
    print(json.dumps({'event':'done','path':str(path),'wall_s':result['wall_s'],
                      'process_elapsed_s':record['process_elapsed_s']}),flush=True)


def paired(directory, case, baseline, candidate):
    for repeat in (1,2):
        run_one(case,baseline,directory/f'base{repeat}.json')
        run_one(case,candidate,directory/f'candidate{repeat}.json')
    result = summarize(directory)
    print(json.dumps({'event':'summary','path':str(directory),**result}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',action='append',default=[])
    parser.add_argument('--stack',action='append',default=[])
    args = parser.parse_args()
    if bool(args.candidate) == bool(args.stack):
        parser.error('choose either --candidate names or --stack patch specs')
    gpu = subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    args.output.mkdir(parents=True,exist_ok=True)
    snapshot = args.output/'gpu_before.txt'
    if snapshot.exists():
        snapshot = args.output/f'gpu_before_{time.time_ns()}.txt'
    snapshot.write_text(gpu.stdout+gpu.stderr)
    if args.candidate:
        for name in args.candidate:
            spec = f'experiments.sam2_profile.patches.{name}:install'
            paired(args.output/name,'han',[ROPE],[ROPE,spec])
    else:
        for case in ('han','felix','bhc'):
            paired(args.output/case,case,[],args.stack)

if __name__ == '__main__':
    main()
