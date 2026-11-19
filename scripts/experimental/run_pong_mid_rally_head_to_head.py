"""Compare unchanged standard and shorter probes on fresh four-second switches."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import signal
import time

import numpy as np
import torch

from fly_connectome.control.session import model_bytes, source_signature
from pong_visual_button_effects import VisualButtonEffects, VisualIntentController
from run_pong_fast_mid_rally_scout import EarlyVisualIntentController
import run_pong_mid_rally_remap as mid
from run_pong_inverse_cycle import save, sha
from run_pong_surprise_recovery import exclusions, reached_deadline, runtime_dependencies, state_hash

PAIRS = 24
START_SEED = 1192742338
WALL_BUDGET = 180
ARMS = {'standard': VisualIntentController, 'shorter': EarlyVisualIntentController}
RUNTIME = (*mid.FILES, 'run_pong_fast_mid_rally_scout.py', 'run_pong_mid_rally_head_to_head.py')


def read(path):
    return json.loads(path.read_text())


def prepare(directory, checkpoint):
    acquired = read(checkpoint)
    receipt = checkpoint.parent / 'live/game-05.json'
    source_receipt = read(receipt)
    assert acquired == source_receipt['state'] and state_hash(acquired) == source_receipt['after']
    assert not acquired['probe_queue'] and acquired['updates_enabled'] and acquired['completed_calibrations'] == 1
    sources, metadata = exclusions(directory.parent)
    sources.add(START_SEED)
    rng = np.random.default_rng(START_SEED)
    seeds = []
    while len(seeds) < PAIRS:
        seed = int(rng.integers(1, mid.MODULUS))
        if seed not in sources and ((seed * 48271 + 1) % mid.MODULUS) % 2 == len(seeds) % 2:
            seeds.append(seed)
            sources.add(seed)
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'schedules.json', seeds)
    save(directory / 'checkpoint-acquired.json', acquired)
    files = [Path(__file__).with_name(name) for name in RUNTIME]
    files.extend((checkpoint, receipt, directory / 'schedules.json', directory / 'checkpoint-acquired.json'))
    spec = dict(contract='fresh-paired-standard-v-short-visual-request-reidentification-v1',
        source=source_signature(), dependencies=runtime_dependencies(),
        files={str(path): sha(path) for path in files}, excluded_prior_metadata=metadata,
        acquired_checkpoint_sha256=state_hash(acquired), starts=PAIRS, start_seed=START_SEED,
        arms=list(ARMS), learners_reloaded_from_same_acquired_checkpoint=True,
        hidden_switch_tick=mid.SWITCH_TICK, terminal_limit_ticks=mid.END_TICK,
        same_perception_and_inherited_intent=True, no_tuning=True, no_extension=True,
        endpoints='all first/later changed opportunities, fully correct ordering before terminal and before first changed opportunity, exact unchanged normal prefix',
        budget=dict(games=2 * PAIRS, transitions=2 * PAIRS * 1000,
                    initials=2 * PAIRS, seconds=WALL_BUDGET, workers=2))
    save(directory / 'spec.json', spec)
    return spec, sha(directory / 'spec.json'), seeds, acquired


def stream(directory, arm, seeds, acquired, deadline):
    signal.signal(signal.SIGALRM, reached_deadline)
    torch.set_num_threads(1)
    mid.VisualIntentController = ARMS[arm]
    directory.mkdir(exist_ok=False)
    active = directory / 'active.json'
    for index, seed in enumerate(seeds):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        learner = VisualButtonEffects.from_state(acquired)
        save(active, dict(index=index, seed=seed, camera_grant=1000, initial_grant=1))
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            row = mid.play(seed, learner)
        except TimeoutError:
            return
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        row.update(index=index, seed=seed, arm=arm, acquired_sha256=state_hash(acquired),
                   probe_frames=learner.dither_count - acquired['dither_count'],
                   final_state_sha256=state_hash(learner.state()))
        save(directory / f'game-{index:02d}.json', row)
        active.unlink()


def run(directory, checkpoint):
    model_bytes()
    spec, digest, seeds, acquired = prepare(directory, checkpoint)
    start = time.monotonic()
    deadline = start + WALL_BUDGET
    save(directory / 'work-clock.json', dict(start=start, deadline=deadline, spec_sha256=digest))
    errors = {}
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:
        futures = {arm: pool.submit(stream, directory / arm, arm, seeds, acquired, deadline) for arm in ARMS}
        for arm, future in futures.items():
            try:
                future.result()
            except Exception as error:
                errors[arm] = repr(error)
    receipts = {arm: {row['index']: row for row in
        (read(path) for path in sorted((directory / arm).glob('game-*.json')))} for arm in ARMS}
    pairs = []
    for index in sorted(set.intersection(*(set(rows) for rows in receipts.values()))):
        a, b = (receipts[arm][index] for arm in ARMS)
        assert a['seed'] == b['seed'] == seeds[index]
        first = lambda row: row['post_change_events'][0] if row['post_change_events'] else None
        left, right = first(a), first(b)
        pairs.append(dict(index=index, seed=a['seed'],
            prefixes_match=(a['prefix_sha256'] == b['prefix_sha256']),
            both_switched=a['entered_switch'] and b['entered_switch'],
            first_clock_match=bool(left and right and left['tick'] == right['tick']),
            first_results=[item['hit'] if item else None for item in (left, right)]))
    records = [row for arm in receipts.values() for row in arm.values()]
    unsaved = sum((directory / arm / 'active.json').exists() for arm in ARMS)
    transitions = sum(row['transitions'] for row in records)
    assert transitions + 1000 * unsaved <= spec['budget']['transitions']
    model_bytes()
    valid_source = (source_signature() == spec['source'] and runtime_dependencies() == spec['dependencies']
        and sha(directory / 'spec.json') == digest
        and all(sha(Path(path)) == expected for path, expected in spec['files'].items()))
    report = dict(complete=len(records) == 2 * PAIRS and not errors, errors=errors, spec_sha256=digest,
        sources_assets_unchanged=valid_source, pairs=len(pairs),
        matched_normal_prefixes=sum(pair['prefixes_match'] for pair in pairs),
        matched_first_changed_clocks=sum(pair['first_clock_match'] for pair in pairs),
        shorter_first_only=sum(pair['first_results'] == [False, True] for pair in pairs),
        standard_first_only=sum(pair['first_results'] == [True, False] for pair in pairs),
        arms={arm: mid.metrics(list(rows.values())) for arm, rows in receipts.items()},
        transitions=transitions, initials=len(records), unsaved_camera_grant=1000 * unsaved,
        wall_seconds=time.monotonic() - start)
    save(directory / 'matched-pairs.json', pairs)
    save(directory / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory.resolve(), args.checkpoint.resolve()), sort_keys=True))
