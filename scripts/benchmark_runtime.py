"""Paired headless/preview producer timing and exact checkpoint equivalence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

from fly_connectome.telemetry import Telemetry
from fly_connectome.training import load_checkpoint


def benchmark(checkpoint, steps, repeats, threads):
    torch.set_num_threads(threads)
    durations = {'headless': [], 'preview': []}
    checkpoint_hashes = set()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for repeat in range(repeats):
            for preview in ([False, True] if repeat % 2 == 0 else [True, False]):
                model = load_checkpoint(checkpoint)
                name = 'preview' if preview else 'headless'
                telemetry = Telemetry(root / f'{repeat}-{name}')
                if preview:
                    telemetry.subscribe()
                start = time.perf_counter()
                next_snapshot = start
                for _ in range(steps):
                    model.step()
                    now = time.perf_counter()
                    if now >= next_snapshot and model.step_index % model.config.sync_steps == 0:
                        if preview:
                            telemetry.subscribe()
                        telemetry.publish(model.snapshot)
                        next_snapshot = now + .5
                durations[name].append(time.perf_counter() - start)
                path = root / 'result.pt'
                model.save(path)
                with path.open('rb') as stream:
                    checkpoint_hashes.add(hashlib.file_digest(stream, 'sha256').hexdigest())
                telemetry.close()
                print(name, durations[name][-1], flush=True)
    overhead = statistics.median(durations['preview']) / statistics.median(durations['headless']) - 1
    return dict(steps=steps, repeats=repeats, threads=threads, torch_version=str(torch.__version__),
                durations_seconds=durations, median_producer_overhead=overhead,
                meets_one_percent_producer_target=overhead <= .01,
                checkpoints_byte_identical=len(checkpoint_hashes) == 1,
                scope='producer timing; browser rendering and server CPU contention are not included')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.threads) < 1:
        parser.error('positive steps/repeats/threads required')
    result = benchmark(args.checkpoint, args.steps, args.repeats, args.threads)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
