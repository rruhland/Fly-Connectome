"""Play Pong through the arm using the unchanged archived visual score strategy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

from arm_plays_pong import MatchCase, NATIVE_HZ, SCREEN_TICKS, run_match
from button_pong_skill import PongScreen, paired_interval, write_json
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_button_panel import SOURCE_GEAR, VisualButtonMotor, pretrain_motor
from pong_risk_aware_placement import build_risk_aware_controller
from pong_score_placement import ScorePlacement
from pong_transfer_arena import interception_policy
from public_entity_pong_native import snapshot


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'artifacts' / 'm1b'
CHECKPOINT = ARCHIVE / '2026-11-19-pong-transfer-checkpoint.json'
COURSE_SEED = 53_057_931
PRACTICE_SEED = 147_884_320
CLOCK_SECONDS = 25
PERIOD_SECONDS = SCREEN_TICKS / NATIVE_HZ
CONDITIONS = ('archived', 'center')


@dataclass(frozen=True)
class PublicPongRaster(PongScreen):
    raster: np.ndarray


def with_raw_camera(read):
    """Attach the same-time pixels already rendered by the shared runner."""
    def decode(frame):
        screen = read(frame)
        return PublicPongRaster(screen.paddle_y, screen.opponent_y, screen.ball, frame[1::3, 1::3])
    return decode


class ArchivedPongIntent:
    """Translate historical virtual game commands into neutral/up/down intent."""

    def __init__(self, checkpoint, *, center):
        self.checkpoint = checkpoint
        self.center = center
        self.vision = load_default()
        self.point_trace = []
        relation = checkpoint['interception']
        inherited = interception_policy()
        if (tuple(relation['choices']) != inherited.original.choices
                or relation['coefficients'] != inherited.original.coefficients.tolist()
                or relation['scales'] != inherited.original.scales.tolist()
                or relation['require_first_hit'] or inherited.require_first_hit
                or relation['unsupported_fallback'] != 1):
            raise ValueError('archived relational policy does not match the current historical builder')
        self.scores = (ScorePlacement(updates_enabled=False) if center else
                       ScorePlacement(updates_enabled=False, **checkpoint['score']))

    def begin(self, seed):
        self.controller = build_risk_aware_controller(
            seed, self.checkpoint['motor'], self.checkpoint['timing'], self.scores,
            timing_updates=False, forecast_center=self.center)
        self.vision.reset_state()
        self.encoder = VisualStateEncoder(64, 64, PERIOD_SECONDS)
        self.previous = torch.zeros((64, 64), dtype=torch.bool)
        self.entities = []
        self.samples = 0

    def action(self, screen, tick):
        frame = torch.from_numpy(screen.raster.copy())
        events = torch.stack((self.previous & ~frame, ~self.previous & frame)).float()
        self.entities = snapshot(self.encoder.encode_entities(*self.vision.track_entities(events, frame.float())))
        self.previous = frame
        if self.samples:
            self._observe(0)
        request, _ = self.controller.choose(self.entities, 1)
        self.samples += 1
        return 1 - request

    def _observe(self, score, *, final=False):
        self.controller.observe(public_entities=self.entities, own_hits=0, reward=score,
                                terminated=final and bool(score), truncated=final and not score,
                                elapsed_seconds=PERIOD_SECONDS)

    def finish(self, score):
        self._observe(score, final=True)
        self.point_trace.append(dict(camera_frames=self.samples, score=score,
                                     attempted_placements=[asdict(choice) for choice in self.controller.placement.choices],
                                     cancelled_placements=self.controller.cancelled,
                                     motor_order=list(self.controller.timing.belief.order)))


def paired_case(job):
    torch.set_num_threads(1)
    case, checkpoint, memory = job
    matched = {}
    for condition in CONDITIONS:
        policy = ArchivedPongIntent(checkpoint, center=condition == 'center')
        motor = VisualButtonMotor(case.controller_seed ^ 0x6D54F893, memory=memory)
        before = policy.scores.to_state()
        # The isolated camera wrapper retains raw public pixels for this match.
        # The arm, contacts, game timing and raster producer are shared unchanged.
        with patch.object(PongScreen, 'read', with_raw_camera(PongScreen.read)):
            result = run_match(case, policy, motor, first_to=3, max_ticks=CLOCK_SECONDS * NATIVE_HZ,
                               opponent_speed=.35, panel_kwargs={'gear': SOURCE_GEAR})
        if before != policy.scores.to_state():
            raise AssertionError('archived scoring evidence changed during the evaluation')
        result['archived_controller_trace'] = policy.point_trace
        matched[condition] = result
    openings = [row['first_incoming'] for row in matched.values()]
    geometry = lambda incoming: None if incoming is None else {k: v for k, v in incoming.items() if k != 'success'}
    if geometry(openings[0]) != geometry(openings[1]):
        raise AssertionError('paired opening trajectories differ')
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = json.loads(CHECKPOINT.read_text())
    rng = np.random.default_rng(COURSE_SEED)
    seeds = rng.choice(50_000_000, size=8, replace=False) + 1_750_000_000
    with ZipFile(ARCHIVE / '2026-11-19-pong-score-placement.zip') as source:
        upstream = {seed for group in json.loads(source.read('schedule.json')).values() for seed in group}
    if set(seeds) & upstream:
        raise ValueError('archived practice and receiving starts overlap')
    wires = rng.permutation([0, 3] * 4)
    cases = [MatchCase(int(seed), int(rng.integers(2**32)), int(wire), int(rng.integers(2**32)))
             for seed, wire in zip(seeds, wires, strict=True)]
    memory, practice = pretrain_motor(PRACTICE_SEED, ticks=80, gear=SOURCE_GEAR)
    write_json(args.out / 'manifest.json', dict(checkpoint=str(CHECKPOINT.relative_to(ROOT)),
               checkpoint_sha256=sha256(CHECKPOINT.read_bytes()).hexdigest(),
               schedule_seed=COURSE_SEED, starts=[asdict(case) for case in cases], arm_practice=practice,
               arm_practice_seed=PRACTICE_SEED, conditions=CONDITIONS, first_to=3, clock_seconds=CLOCK_SECONDS,
               opponent_speed=.35, gear=SOURCE_GEAR, archived_playback_fps=1 / PERIOD_SECONDS,
               native_archive_camera_fps=1 / checkpoint['camera']['seconds'],
               raw_camera='native 192-pixel frame sampled every fourth tick, centered decimation to 64 pixels',
               source_code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
               frozen_archived_fields=['score', 'interception'], restored_point_priors=['motor', 'timing']))
    write_json(args.out / 'arm-motor.json', memory)
    rows = {condition: [] for condition in CONDITIONS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(case, checkpoint, memory) for case in cases]
        for matched in pool.map(paired_case, jobs):
            for condition, row in matched.items():
                rows[condition].append(row)
            write_json(args.out / 'matches.json', rows)
            print('completed paired matches', len(rows['archived']), flush=True)
    keys = ('own_returns', 'own_opportunities', 'scored_after_own_return', 'points_won', 'points_lost', 'seconds')
    summary = {condition: {key: sum(row[key] for row in group) for key in keys}
               for condition, group in rows.items()}
    for condition, group in rows.items():
        summary[condition].update(own_match_wins=sum(row['winner'] == 'own' for row in group),
                                  opponent_match_wins=sum(row['winner'] == 'opponent' for row in group),
                                  capped_matches=sum(row['winner'] is None for row in group))
        for row in group:
            row['seed'] = row['case']['pong_seed']
    summary['paired_goal_difference'] = paired_interval(rows['archived'], rows['center'],
                                                         lambda row: row['scored_after_own_return'])
    write_json(args.out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
