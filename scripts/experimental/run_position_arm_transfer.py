"""Transfer carriage reward knowledge into a redundant three-joint position servo."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

from body_strategy_transfer import run_experiment
from planar_position_arm import ACTION_DIM, Arena


def position_arm(body, case, *, max_steps):
    if body != 'three_joint_position_arm':
        raise ValueError('expected a three-joint position arm')
    # The starting posture varies independently of the rewarded target and policy.
    start = struct.pack('!dd', *case.start)
    seed = int.from_bytes(sha256(start).digest()[:8], 'big')
    return Arena(case, posture_seed=seed, max_steps=max_steps)


def run(first_seed, streams):
    result = run_experiment(range(first_seed, first_seed + streams),
                            target_body='three_joint_position_arm',
                            target_action_dim=ACTION_DIM, target_arena_factory=position_arm)
    for name in ('planar_position_arm.py', 'run_position_arm_transfer.py'):
        result['source_sha256'][name] = sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--first-seed', type=int, default=86400)
    parser.add_argument('--streams', type=int, default=64)
    args = parser.parse_args()
    data = run(args.first_seed, args.streams)
    with args.output.open('x') as out:
        json.dump(data, out, separators=(',', ':'), allow_nan=False)
    print(data['summary']['first'])
    print(data['paired_first_ticks_saved'])
