"""Reproducible headless entrypoints; UI is a separate optional process."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    native = commands.add_parser('build-native', help='build optional strict-float B=1 CPU kernels with g++')
    native.add_argument('--output', required=True)
    native.add_argument('--compiler', default='g++')
    download = commands.add_parser('download', help='download and verify pinned MaleCNS sources')
    download.add_argument('--manifest', default='data/malecns-v1.0-sources.json')
    download.add_argument('--directory', default='data/raw')
    init = commands.add_parser('init', help='initialize a measured graph checkpoint')
    init.add_argument('--artifacts', default='data/cache/milestone-1')
    init.add_argument('--stage', choices=['M1A', 'M1B'], default='M1A')
    init.add_argument('--threshold', type=int, choices=[1, 3, 5])
    init.add_argument('--seeds', default='1')
    init.add_argument('--warm')
    init.add_argument('--output', required=True)
    init.add_argument('--device', default='cpu')
    init.add_argument('--profile', help='versioned JSON neuron/sensory profile; omission preserves legacy dynamics')
    train = commands.add_parser('train', help='resume training and save exact state')
    train.add_argument('checkpoint')
    train.add_argument('--steps', type=int, required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--device', default='cpu')
    train.add_argument('--threads', type=int)
    train.add_argument('--checkpoint-every', type=int, default=10000)
    train.add_argument('--native-library', help='explicit compiled B=1 CPU kernel library')
    evaluate = commands.add_parser('evaluate', help='frozen comparisons on held-out seeds')
    evaluate.add_argument('checkpoint')
    evaluate.add_argument('--initial', required=True)
    evaluate.add_argument('--seeds', default='1001,1002,1003')
    evaluate.add_argument('--steps', type=int, required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--threads', type=int)
    diagnose = commands.add_parser('diagnose', help='offline visual probes against a frozen checkpoint')
    diagnose.add_argument('checkpoint')
    diagnose.add_argument('--probe-json')
    diagnose.add_argument('--silence', default='')
    diagnose.add_argument('--seed', type=int, default=1001)
    diagnose.add_argument('--output', required=True)
    diagnose.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.command == 'build-native':
        from .native_cpu import build_library
        build_library(args.output, args.compiler)
        print(json.dumps(dict(library=args.output)))
        return
    if args.command == 'download':
        from .data import download_sources
        download_sources(args.manifest, args.directory)
        print('Pinned sources verified.')
        return
    if args.command == 'init':
        from .artifacts import initialize
        model = initialize(args.artifacts, stage=args.stage, seeds=[int(x) for x in args.seeds.split(',')],
                           threshold=args.threshold, device=args.device, warm_checkpoint=args.warm,
                           dynamics_profile=json.loads(Path(args.profile).read_text()) if args.profile else None)
        model.save(args.output)
        print(json.dumps(dict(checkpoint=args.output, neurons=model.network.n, edges=model.network.e,
                              stage=args.stage, graph_sha256=model.network.graph.identity())))
    elif args.command == 'train':
        import torch
        from .training import load_checkpoint
        if args.steps < 1 or args.checkpoint_every < 1 or (args.threads is not None and args.threads < 1):
            parser.error('steps, threads and checkpoint interval must be positive')
        if args.threads is not None:
            torch.set_num_threads(args.threads)
        model = load_checkpoint(args.checkpoint, device=args.device)
        if args.native_library:
            from .native_cpu import NativeCPU
            NativeCPU(args.native_library).enable(model)
        stopped = False
        def request_stop(*_):
            nonlocal stopped
            stopped = True
        previous_handler = signal.signal(signal.SIGINT, request_stop)
        start = time.perf_counter()
        completed = 0
        try:
            for _ in range(args.steps):
                model.step()
                completed += 1
                if completed % args.checkpoint_every == 0 and model.step_index % model.config.sync_steps == 0:
                    model.save(args.output)
                    print(json.dumps(dict(event='checkpoint',checkpoint=args.output,step=model.step_index,
                                          elapsed_seconds=time.perf_counter()-start)),flush=True)
                if stopped and model.step_index % model.config.sync_steps == 0:
                    break
            model.save(args.output)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        print(json.dumps(dict(checkpoint=args.output, completed_steps=completed, metrics=model.metrics,
                              backend='native-cpu' if args.native_library else 'torch',
                              steps_per_second=completed / (time.perf_counter() - start))))
    elif args.command == 'evaluate':
        import torch
        if args.threads is not None:
            if args.threads < 1:
                parser.error('threads must be positive')
            torch.set_num_threads(args.threads)
        from .evaluation import compare
        report = compare(args.checkpoint, args.initial, [int(x) for x in args.seeds.split(',')], args.steps, args.device)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    elif args.command == 'diagnose':
        import torch
        from .diagnostics import Probe, run_probe
        probe = Probe(**json.loads(Path(args.probe_json).read_text())) if args.probe_json else Probe()
        result = run_probe(args.checkpoint, probe, seed=args.seed, device=args.device,
                           silenced_types=[x.strip() for x in args.silence.split(',') if x.strip()])
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        print(json.dumps(dict(bundle=str(path), stimulus=asdict(probe), checkpoint_sha256=result['checkpoint_sha256'])))


if __name__ == '__main__':
    main()
