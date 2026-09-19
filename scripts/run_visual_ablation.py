"""Execute one preregistered M1A ablation with resumable checkpoints."""
import argparse
import json
from pathlib import Path
import time

import torch

from fly_connectome.artifacts import initialize
from fly_connectome.data import checksum
from fly_connectome.training import load_checkpoint
from audit_prediction import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition')
    parser.add_argument('--spec',default='configs/event-learning-ablation-v1.json')
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    condition = next(c for c in spec['conditions'] if c['name'] == args.condition)
    profile = json.loads(Path(condition['profile']).read_text())
    prefix = f'event-v1-{args.condition}'
    initial, trained = (Path(f'checkpoints/{prefix}-{suffix}.pt') for suffix in ('initial','trained'))
    torch.set_num_threads(1)
    if trained.exists():
        model = load_checkpoint(trained)
    elif initial.exists():
        model = load_checkpoint(initial)
    else:
        model = initialize('data/cache/milestone-1',seeds=[spec['training_seed']],dynamics_profile=profile)
        model.save(initial)
    if model.manifest.get('dynamics_profile') != profile:
        raise ValueError('refusing to resume an ablation with a different profile')
    if model.seeds != [spec['training_seed']] or model.step_index > spec['training_steps']:
        raise ValueError('refusing to resume mismatched ablation seeds or training length')
    start = time.perf_counter()
    while model.step_index < spec['training_steps']:
        model.step()
        if model.step_index%100 == 0:
            model.save(trained)
            print(json.dumps(dict(condition=args.condition,step=model.step_index,elapsed_seconds=time.perf_counter()-start)),flush=True)
    model.save(trained)
    del model
    report = dict(spec_sha256=checksum(args.spec),profile_sha256=checksum(condition['profile']))
    for label,path in (('initial',initial),('trained',trained)):
        report[label] = audit(str(path),spec['development_steps'],spec['development_seeds'])
        Path(f'runs/{prefix}-audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(condition=args.condition,evaluation=label,events=report[label]['events'])),flush=True)


if __name__ == '__main__':
    main()
