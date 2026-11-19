"""Development-only within-rally scout of short, drift-compensated held requests."""

import argparse
import json
from pathlib import Path
import signal
import time

import torch

from fly_connectome.control.session import model_bytes, source_signature
from pong_visual_button_effects import VisualButtonEffects, VisualIntentController
import run_pong_mid_rally_remap as mid
from run_pong_inverse_cycle import save, sha
from run_pong_surprise_recovery import reached_deadline, state_hash

EARLY_CAMERAS = 4
TOTAL_CAMERAS = 12
WALL_BUDGET = 120


class EarlyVisualIntentController(VisualIntentController):
    """Measure early and later slopes; subtract their common inertial carry."""

    def _reset_dwell(self):
        super()._reset_dwell()
        self.probe_start = self.probe_early = None

    def choose(self, entities, baseline_action):
        if (self.learner.updates_enabled and self.learner.recalibration_requested
                and not self.probe_attempted_here and self._motor_position(entities) is not None):
            self.learner.start_probes()
            self.probe_attempted_here = True
        return super().choose(entities, baseline_action)

    def observe(self, *, own_hits, reward, terminated, truncated, elapsed_seconds, public_entities):
        if not self.probe_active:
            return super().observe(own_hits=own_hits, reward=reward, terminated=terminated,
                truncated=truncated, elapsed_seconds=elapsed_seconds, public_entities=public_entities)
        action, _, before = self.pending
        self.pending = None
        after = self._motor_position(public_entities)
        self.inherited.observe(own_hits, terminated)
        if before is None or after is None:
            self._reset_dwell()
            return
        if action != self.last_action:
            self._reset_dwell()
            self.last_action = action
            self.probe_start = before
        self.dwell += 1
        if self.dwell == EARLY_CAMERAS:
            self.probe_early = after
        if self.dwell == TOTAL_CAMERAS:
            early = self.probe_early - self.probe_start
            later = after - self.probe_early
            # A coasting positive slope decays. Reward increased positive
            # movement over a decaying continuation of the first interval.
            self.learner.record_probe(1.5 * (later - early))
            self._reset_dwell()


def read(path):
    return json.loads(path.read_text())


def run(directory, baseline):
    model_bytes()
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM, reached_deadline)
    mid.VisualIntentController = EarlyVisualIntentController
    starts = read(baseline / 'schedules.json')
    acquired = read(baseline / 'checkpoint-acquired.json')
    original = [read(baseline / f'pair-{i:02d}-live.json') for i in range(len(starts))]
    assert len(starts) == 12 and all(row['seed'] == seed for row, seed in zip(original, starts, strict=True))
    files = [Path(__file__), Path(mid.__file__), Path(__file__).with_name('pong_visual_button_effects.py'),
        baseline / 'summary.json', baseline / 'spec.json', baseline / 'schedules.json', baseline / 'checkpoint-acquired.json']
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'schedules.json', starts)
    spec = dict(contract='short-visual-button-probe-mid-rally-development-v1',
        source=source_signature(), development_reuse='same 12 direct-test starts and acquired checkpoint',
        files={str(path): sha(path) for path in files}, starts=len(starts), max_transitions=12000,
        max_initial=12, wall_budget_seconds=WALL_BUDGET, early_cameras=EARLY_CAMERAS,
        total_cameras=TOTAL_CAMERAS, start_radius=None, unchanged_trigger=True,
        response='1.5 times displacement after the first four cameras minus 1.5 times displacement during those four cameras',
        pass_scope='descriptive development only; one fixed run, no tuning or extension')
    save(directory / 'spec.json', spec)
    digest = sha(directory / 'spec.json')
    began = time.monotonic()
    deadline = began + WALL_BUDGET
    rows = []
    active = directory / 'active.json'
    timed_out = False
    try:
        for index, seed in enumerate(starts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('wall budget reached at game dispatch')
            learner = VisualButtonEffects.from_state(acquired)
            save(active, dict(seed=seed, index=index, transition_grant=1000, initial_grant=1))
            signal.setitimer(signal.ITIMER_REAL, remaining)
            try:
                row = mid.play(seed, learner)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            row.update(index=index, seed=seed, same_prefix_as_baseline=
                row['prefix_sha256'] == original[index]['prefix_sha256'],
                probe_frames=learner.dither_count - acquired['dither_count'],
                acquired_sha256=state_hash(acquired), final_sha256=state_hash(learner.state()))
            save(directory / f'game-{index:02d}.json', row)
            active.unlink()
            rows.append(row)
            print(index + 1, row['first_correct_order_seconds'], row['same_prefix_as_baseline'], flush=True)
    except TimeoutError:
        timed_out = True
    model_bytes()
    result = dict(complete=len(rows) == len(starts) and not timed_out,
        timed_out=timed_out, metrics=mid.metrics(rows), baseline_live=mid.metrics(original),
        prefix_matched=sum(r['same_prefix_as_baseline'] for r in rows),
        transitions=sum(r['transitions'] for r in rows), initials=sum(r['initial'] for r in rows),
        unsaved_camera_grant=1000 if active.exists() else 0,
        wall_seconds=time.monotonic() - began, spec_sha256=digest,
        sources_unchanged=(spec['source'] == source_signature()
            and sha(directory / 'spec.json') == digest
            and all(sha(Path(p)) == expected for p, expected in spec['files'].items())))
    save(directory / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('baseline', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory.resolve(), args.baseline.resolve()), sort_keys=True))
