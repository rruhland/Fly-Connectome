"""Passive credit recording during bounded exact replay and visual challenges."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from frame_prediction import FramePrediction
from fly_connectome.plasticity import LearningConfig
from fly_connectome.sensor import Retina, EventCamera
from fly_connectome.data import checksum
from controlled_visual import make_network
from multitempo import training_schedule
from timing_transfer import load_model


class CreditRecorder(FramePrediction):
    state_names = ['sensory_state', 'predictive_current', 'excitatory_prediction',
                   'inhibitory_prediction', 'voltage', 'adaptation']

    def __init__(self, *args, audit_edges, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_edges = torch.as_tensor(audit_edges, dtype=torch.long)
        self.rows = []
        self.issue_context = None

    def state(self):
        n = self.network
        post, pre = n.post[self.audit_edges], n.pre[self.audit_edges]
        columns = [getattr(n, name)[0, post] for name in self.state_names]
        columns += [n.voltage[0, pre], n.adaptation[0, pre]]
        return torch.stack(columns, dim=1).numpy().copy()

    def selected(self, edges, values):
        result = torch.zeros(len(self.audit_edges), dtype=values.dtype)
        for j, edge in enumerate(self.audit_edges):
            match = edges == edge
            if match.any():
                result[j] = values[match].sum()
        return result

    def _capture_forecast(self):
        super()._capture_forecast()
        keys, eligibility, prediction = self.forecast
        self.issue_context = dict(issue_tick=self.tick, issue_state=self.state(),
            eligibility=self.selected(keys, eligibility).numpy(),
            prediction=self.selected(keys, prediction).numpy(),
            issue_weights=self.network.magnitudes[self.audit_edges].numpy().copy())

    @torch.no_grad()
    def observe(self, activity, reward):
        due = self.tick % 8 == 0 and self.forecast is not None
        if due:
            row = dict(self.issue_context, confirm_tick=self.tick, confirm_state=self.state())
            before = self.proposals[self.audit_edges].clone()
        super().observe(activity, reward)
        if due:
            edges, target, delta = self.last_visual_update
            selected = self.selected(edges, delta)
            actual = self.proposals[self.audit_edges]-before
            torch.testing.assert_close(actual, selected, rtol=0, atol=0)
            row.update(target=self.selected(edges, target).numpy(), delta=selected.numpy(),
                       accumulator_error=float((actual-selected).abs().max()))
            self.rows.append(row)


def challenge_frames(kind, blanks, dwells, first, second):
    frames, trials, tempos, segments = [], [], [], []
    for trial, (blank, dwell) in enumerate(zip(blanks, dwells)):
        sequence = torch.zeros(int(blank+8*dwell+1), 1, 32, 64, dtype=torch.bool)
        phase = np.full(len(sequence), -1)
        for slot in range(8):
            start = int(blank+slot*dwell)
            phase[start:start+dwell] = slot
            if kind == 'repeat':
                active = [(slot % 4 in [0, 1], first), (slot % 4 in [1, 2], second)]
            else:
                position = (slot % 2 == 0) and not (kind == 'omitted' and slot == 2)
                active = [(True, (30, 39 if position else 38))]
            for enabled, (y, x) in active:
                if enabled:
                    sequence[start:start+dwell, 0, y, x] = True
        frames.append(sequence)
        trials.extend([trial]*len(sequence));tempos.extend([dwell]*len(sequence));segments.extend(phase)
    return torch.cat(frames), dict(trial=np.asarray(trials), dwell=np.asarray(tempos), slot=np.asarray(segments))


@torch.no_grad()
def record(net, crop, metadata, frames, edges):
    cfg = LearningConfig(**metadata['learning'])
    retina = Retina(**crop['retina']);camera = EventCamera(1, 32, 64)
    rule = CreditRecorder(net, cfg, audit_edges=edges, sensory_mask=retina.injected,
                          sensory_gain=metadata['config']['sensory_gain'])
    target = int(net.post[edges[0]])
    predictions, targets, spikes = [], [], []
    begun = time.perf_counter()
    for frame in frames:
        injection = retina.project(camera.observe(frame))*metadata['config']['sensory_gain']
        for tick in range(8):
            a = net.step(injection if tick == 0 else torch.zeros_like(injection), capture_increments=True)
            predictions.append(float(cfg.encode(a.predicted[0, target], net.config.threshold)))
            targets.append(float(cfg.observation(a, net.config.threshold, retina.injected,
                            metadata['config']['sensory_gain'])[0, target]))
            spikes.append(a.spikes[0].numpy().copy())
            rule.observe(a, torch.zeros(1))
        before = net.magnitudes[edges].clone()
        expected = (before+rule.proposals[edges]).clamp(0, cfg.maximum_weight)*torch.exp(-rule.homeostatic_exponent[edges])
        rule.synchronize()
        torch.testing.assert_close(net.magnitudes[edges], expected, rtol=0, atol=0)
    rule.synchronize()
    assert torch.isfinite(net.magnitudes).all()
    result = {k: np.asarray([row[k] for row in rule.rows]) for k in rule.rows[0]}
    result.update(all_prediction=np.asarray(predictions, dtype=np.float32)[:, None],
                  all_target=np.asarray(targets, dtype=np.float32)[:, None],
                  all_spikes=np.asarray(spikes), weights_trained=net.magnitudes.numpy())
    return result, time.perf_counter()-begun


def summarize(r, labels):
    rows = {}
    for e, pre in enumerate([20655, 26550]):
        q, y = r['delta'][:, e], r['target'][:, e]
        predicted = (-1 if e == 0 else 1)*np.where(r['issue_state'][:, e, 0] > 0, 1, -1)
        out = {}
        for label, mask in labels.items():
            out[label] = {}
            for name, category in [('ON', y < 0), ('OFF', y > 0), ('quiet', y == 0)]:
                keep = mask & category & (q != 0)
                count = int(keep.sum())
                out[label][name] = dict(count=count, correct=int((np.sign(q[keep]) == predicted[keep]).sum()),
                    positive=int((q[keep] > 0).sum()), negative=int((q[keep] < 0).sum()),
                    update_sum=float(q[keep].astype(float).sum()),
                    absolute_update_sum=float(np.abs(q[keep]).astype(float).sum()))
        rows[str(pre)] = out
    return rows


def main():
    torch.set_num_threads(1)
    out = Path('runs/credit-replay-v1');out.mkdir(exist_ok=True)
    result_path = out/'results.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    runner_hash = checksum(Path(__file__))
    crop, m, _ = load_model()
    assert m['config']['sync_steps'] == 1 and m['config']['neural_steps'] == 8
    g = crop['graph'];target = int(np.searchsorted(g.body_ids, 82450))
    edges = [int(np.flatnonzero((g.post == target) & (g.body_ids[g.pre] == pre))[0]) for pre in [20655, 26550]]
    retina = Retina(**crop['retina']);column = int(retina.pixel_bins[30*64+39])
    candidates = np.flatnonzero(retina.pixel_bins.numpy() == column)
    candidates = candidates[candidates != 30*64+39]
    assert len(candidates), 'repeat challenge requires two pixels in the existing column'
    other = min(candidates, key=lambda p: abs(p//64-30)+abs(p%64-39))
    second = (int(other//64), int(other%64))
    full, blanks, dwells = training_schedule(200, 9060)
    prefix_length = int((blanks[:24]+8*dwells[:24]+1).sum())
    rng = np.random.default_rng(9091)
    challenge_dwells = np.tile([2, 3, 4, 6], 4);rng.shuffle(challenge_dwells)
    challenge_blanks = rng.integers(12, 37, size=16)
    for name, source, kinetics in [
        ('baseline', Path('runs/multitempo-v1/mixed-training.npz'), 'area-matched-excitation-v1'),
        ('F', Path('runs/short-term-v1/F-training.npz'), 'short-term-F-v1')]:
        old = np.load(source);source_hash = checksum(source)
        for kind in ['replay', 'standard', 'omitted', 'repeat']:
            key = f'{name}-{kind}';path = out/f'{key}.npz'
            if kind == 'replay':
                frames = full[:prefix_length]
                info = dict(trial=np.repeat(np.arange(24), blanks[:24]+8*dwells[:24]+1),
                            dwell=np.repeat(dwells[:24], blanks[:24]+8*dwells[:24]+1))
                weights = crop['weights']
            else:
                frames, info = challenge_frames(kind, challenge_blanks, challenge_dwells, (30, 39), second)
                weights = torch.from_numpy(old['weights_trained'])
            stimulus_hash = hashlib.sha256(frames.numpy().tobytes()).hexdigest()
            if key in results:
                saved = results[key]
                assert saved['runner_sha256'] == runner_hash and saved['source_sha256'] == source_hash
                assert saved['stimulus_sha256'] == stimulus_hash and saved['sha256'] == checksum(path)
                print('Verified saved', key, flush=True)
                continue
            net = make_network(crop, m, weights, predictive_kinetics=kinetics)
            r, seconds = record(net, crop, m, frames, edges)
            if kind == 'replay':
                for field in ['prediction', 'target', 'spikes']:
                    np.testing.assert_array_equal(r['all_'+field], old[field][:len(frames)*8])
                selected = [int(np.flatnonzero(old['incoming_edges'] == e)[0]) for e in edges]
                np.testing.assert_array_equal(r['eligibility'], old['eligibility'][r['issue_tick']+1][:, selected])
            else:
                initial = net.magnitudes.new_tensor(old['weights_trained'])
                assert not torch.equal(net.magnitudes, initial)
            confirm = r['confirm_tick']//8
            labels = {'all': np.ones(len(confirm), dtype=bool),
                      'early': info['trial'][confirm] < len(np.unique(info['trial']))//2,
                      'late': info['trial'][confirm] >= len(np.unique(info['trial']))//2}
            labels.update({f'dwell{d}': info['dwell'][confirm] == d for d in np.unique(info['dwell'])})
            np.savez_compressed(path, **r, **{f'frame_{k}': v for k, v in info.items()})
            results[key] = dict(seconds=seconds, frames=len(frames), source_sha256=source_hash,
                runner_sha256=runner_hash, stimulus_sha256=stimulus_hash, sha256=checksum(path),
                second_pixel=second, exact_replay=(kind == 'replay'),
                accumulator_max_error=float(r['accumulator_error'].max()),
                incoming_weights_initial=weights[edges].tolist(),
                incoming_weights_final=net.magnitudes[edges].tolist(), metrics=summarize(r, labels))
            result_path.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
            print(key, 'complete', round(seconds, 2), 'seconds', flush=True)


if __name__ == '__main__':
    main()
