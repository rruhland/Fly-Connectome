"""Read-only checkpoint probe of exact neuron inactivity; instrumentation is not a speed benchmark."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from fly_connectome.data import checksum
from fly_connectome.native_cpu import NativeCPU
from fly_connectome.training import load_checkpoint


@torch.no_grad()
def profile(checkpoint, library, frames, native_threads):
    torch.set_num_threads(1)
    before = checksum(checkpoint)
    model = load_checkpoint(checkpoint)
    model.config = replace(model.config, metrics_mode='events')
    kernel = NativeCPU(library, threads=native_threads)
    kernel.enable(model)
    net = model.network
    state_names = ('voltage', 'feedforward_current', 'predictive_current',
                   'behavioral_current', 'sensory_state', 'adaptation', 'refractory')
    counts = {}
    ticks = 0
    previous_fixed = None
    original = net.step

    def record(name, mask):
        if name not in counts:
            counts[name] = torch.zeros(net.n, dtype=torch.int64)
        counts[name].add_(mask.flatten())

    def measured(injection, **kwargs):
        nonlocal previous_fixed
        before_states = {name: getattr(net, name).clone() for name in state_names}
        nonzero = torch.zeros_like(net.voltage, dtype=torch.bool)
        for name in state_names:
            mask = getattr(net, name) != 0
            record('pre_nonzero_' + name, mask)
            nonzero |= mask
        rest = net.rest_current[None, :] != 0
        record('pre_nonzero_any_state', nonzero)
        record('rest_driven', rest)
        activity = original(injection, **kwargs)
        arrivals = torch.zeros_like(activity.spikes)
        arrivals[activity.arrival_environments, net.post[activity.arrival_edges]] = True
        injected = injection != 0
        external = arrivals | injected
        unchanged = torch.ones_like(activity.spikes)
        for name, before_state in before_states.items():
            after_state = getattr(net, name)
            if before_state.dtype == torch.float32:
                unchanged &= before_state.view(torch.int32) == after_state.view(torch.int32)
            else:
                unchanged &= before_state == after_state
        fixed = unchanged & ~external & ~activity.spikes
        record('no_input_exact_fixed_point', fixed)
        record('no_input_nonzero_exact_fixed_point', fixed & (nonzero | rest))
        if previous_fixed is not None:
            continuation = previous_fixed & ~external
            record('previous_fixed_no_input_next_tick', continuation)
            record('previous_fixed_still_fixed_next_tick', continuation & fixed)
            record('previous_fixed_lost_without_input', continuation & ~fixed)
        previous_fixed = fixed
        quiescent = ~nonzero & ~rest & ~external
        record('receiving_arrival', arrivals)
        record('receiving_injection', injected)
        record('receiving_arrival_or_injection', external)
        record('exact_quiescent', quiescent)
        record('no_external_but_state_or_rest', ~external & (nonzero | rest))
        record('spiking', activity.spikes)
        record('spiking_without_arrival_or_injection', activity.spikes & ~external)
        assert not (activity.spikes & quiescent).any()
        nonlocal ticks
        ticks += 1
        return activity

    net.step = measured
    start = time.perf_counter()
    model.run(frames)
    seconds = time.perf_counter() - start
    after = checksum(checkpoint)
    assert before == after

    def summarize(indices):
        return {name: {'neuron_ticks': int(values[indices].sum()),
                       'fraction': float(values[indices].sum()) / (ticks * len(indices)),
                       'neurons_ever': int((values[indices] > 0).sum()),
                       'neurons_every_tick': int((values[indices] == ticks).sum())}
                for name, values in counts.items()}

    classes = model.retina.spec['cell_types'] or ['unannotated'] * net.n
    groups = {}
    for index, name in enumerate(classes):
        groups.setdefault(name, []).append(index)
    return dict(checkpoint=str(checkpoint), checkpoint_sha256_before=before,
                checkpoint_sha256_after=after, native_library=str(library),
                native_library_sha256=checksum(library), torch_threads=1,
                native_threads=native_threads, frames=frames, neural_ticks=ticks,
                neurons=net.n, edges=net.e, instrumented_seconds=seconds,
                neuron_config=asdict(net.config), run_config=asdict(model.config),
                metrics=summarize(list(range(net.n))),
                classes={name: {'neurons': len(indices), 'metrics': summarize(indices)}
                         for name, indices in sorted(groups.items())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--native-library', required=True)
    parser.add_argument('--native-threads', type=int, default=4)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    report = profile(args.checkpoint, args.native_library, args.frames, args.native_threads)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'classes'}, indent=2))
