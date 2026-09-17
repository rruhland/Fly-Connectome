"""Independent training/evaluation process; UI lifetime does not own this worker."""
import argparse
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch

from .sensor import Retina
from .telemetry import Telemetry
from .training import Trainer, load_checkpoint


def apply_command(model, state, command, directory):
    action = command['action']
    if action in ('save', 'restart') and model.evaluation:
        raise ValueError("evaluation is read-only")
    if action == 'pause':
        state['paused'] = True
    elif action == 'resume':
        state['paused'] = False
    elif action == 'stop':
        if not model.evaluation:
            model.save(Path(directory) / 'checkpoint-latest.pt')
        state['stop'] = True
    elif action == 'save':
        model.save(Path(directory) / 'checkpoint-latest.pt')
    elif action == 'reset':
        model.reset_environment()
    elif action == 'restart':
        if command.get('confirmed') is not True:
            raise ValueError("restart requires explicit confirmation")
        n = model.network
        fresh = Trainer(n.graph, n.delays.tolist(),
            [('feedforward', 'predictive', 'behavioral')[p] for p in n.pathways.tolist()],
            Retina(**model.retina.spec, device=model.device), model.motor_up, model.motor_down, model.seeds,
            config=model.config, physics=model.environment.config, neurons=n.config,
            learning=model.learning_config, curriculum=model.curriculum, manifest=model.manifest, device=model.device)
        model.__dict__.update(fresh.__dict__)
    elif action == 'step' and model.evaluation:
        state['single_step'] = True
        state['paused'] = True
    elif action == 'control' and model.evaluation:
        if command.get('value') not in ('learned', 'random', 'scripted', 'human'):
            raise ValueError("invalid evaluation control")
        state['control'] = command['value']
    elif action == 'human' and model.evaluation:
        state['human'] = max(-1., min(1., float(command['value'])))
    elif action == 'speed' and model.evaluation:
        state['delay'] = max(0., min(.5, float(command['value'])))
    else:
        raise ValueError("unsupported command for this worker")


def _worker_lock(directory):
    stream = (directory / 'worker.lock').open('a+b')
    stream.seek(0)
    stream.write(b'1')
    stream.flush()
    stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def run_worker(checkpoint, directory, *, evaluation=False, device='cpu', steps=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = _worker_lock(directory)
    model = load_checkpoint(checkpoint, evaluation=evaluation, device=device)
    telemetry = Telemetry(directory)
    state = dict(paused=False, stop=False, single_step=False, control='learned', human=0., delay=.02 if evaluation else 0.)
    logs = deque(maxlen=24)
    started = last_poll = last_publish = time.monotonic()
    initial_step = model.step_index
    try:
        while not state['stop'] and (steps is None or model.step_index - initial_step < steps):
            now = time.monotonic()
            boundary = evaluation or model.step_index % model.config.sync_steps == 0
            if boundary and now - last_poll >= .2:
                last_poll = now
                for path in sorted(directory.glob('command-*.json'))[:16]:
                    try:
                        command = json.loads(path.read_text())
                        if command['action'] == 'probe' and evaluation:
                            from .diagnostics import Probe, run_probe
                            result = run_probe(checkpoint, Probe(**command.get('probe', {})),
                                silenced_types=command.get('silenced_types', []), device=device)
                            torch.save(result, directory / 'diagnostic.pt')
                            summary = {k: {x: v.tolist() if isinstance(v, torch.Tensor) else v for x, v in row.items()}
                                       for k, row in result['populations'].items()}
                            (directory / 'diagnostic.json').write_text(json.dumps(summary))
                        else:
                            apply_command(model, state, command, directory)
                        logs.append(f"Applied {command['action']}")
                    except (ValueError, KeyError) as error:
                        logs.append(str(error))
                    finally:
                        path.unlink(missing_ok=True)
            if state['stop']:
                break
            if not state['paused'] or state['single_step']:
                human = torch.full((model.environment.batch,), state['human'], device=device)
                model.step(state['control'], human)
                state['single_step'] = False
                if state['delay']:
                    time.sleep(state['delay'])
            else:
                time.sleep(.02)
            if (evaluation or model.step_index % model.config.sync_steps == 0) and now - last_publish >= .5:
                last_publish = now
                def capture():
                    snapshot = model.snapshot()
                    snapshot.update(paused=state['paused'], logs=list(logs), pid=os.getpid(),
                        steps_per_second=(model.step_index - initial_step) / max(.001, time.monotonic() - started))
                    return snapshot
                telemetry.publish(capture)
        if not evaluation:
            model.save(directory / 'checkpoint-latest.pt')
        telemetry.publish(lambda: dict(model.snapshot(), stopped=True, paused=False,
                                       logs=list(logs) + ['Worker stopped; final checkpoint saved.' if not evaluation else 'Evaluation stopped.']))
    finally:
        telemetry.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--directory', required=True)
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    run_worker(args.checkpoint, args.directory, evaluation=args.evaluation, device=args.device, steps=args.steps)
