"""Identify motor wiring during the first scored point of a fixed serve stream."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt20_motor_carry import WIDTH, Wiring, descriptor, optical_loss
from .tt20_motor_carry_trial import installation
from .tt22_learned_body import INSTALLATION, SOURCE, native_station, recode, summary
from .tt22_opponent_learn import JitteredOpening, PACE, SPREAD
from .tt22_rally_trial import CONFIG, OUTPUT, alternating


MODES = ('online', 'frozen')
MODEL_SEED = 758299537
TAP_TICKS = 2
TAP_RATE = .45
MIN_GAP = .03
MAX_LOSS = .35
TAP_FRAMES = WIDTH*TAP_TICKS


def pulse(axis, value=TAP_RATE):
    action = np.zeros(WIDTH+1)
    action[axis] = value
    return action


def public_probe(tick):
    return pulse(tick//TAP_TICKS)


class WristFit:
    """Match each observed tap against signed canonical continuations."""

    def __init__(self):
        self.model = JitteredOpening(**dict(CONFIG, view_mode='wrist'))
        self.start_image = None
        self.order, self.signs, self.fits, self.rgb = [], [], [], {}
        self.trials, self.model_frames = 0, 0
        self.seconds = 0.

    def close(self):
        self.model.close()

    def model_prediction(self, axis, sign):
        model = self.model
        model.reset(seed=MODEL_SEED, server=-1, render=False)
        for prior_axis, prior_sign in zip(self.order, self.signs, strict=True):
            preceding = pulse(prior_axis, prior_sign*TAP_RATE)
            for _ in range(TAP_TICKS):
                model.step(preceding, render=False)
        before = model.render_wrist()
        action = pulse(axis, sign*TAP_RATE)
        for _ in range(TAP_TICKS):
            model.step(action, render=False)
        self.model_frames += TAP_TICKS*(1+len(self.order))
        self.trials += 1
        return descriptor((before, model.render_wrist()))

    def observe(self, tick, image):
        self.rgb[str(tick)] = sha256(image).hexdigest()
        if self.start_image is not None:
            clock = perf_counter()
            seen = descriptor((self.start_image, image))
            candidates = sorted((optical_loss(seen, self.model_prediction(axis, sign), 1), axis, sign)
                                for axis in range(WIDTH) if axis not in self.order for sign in (-1, 1))
            loss, axis, sign = candidates[0]
            gap = candidates[1][0]-loss
            self.fits.append(dict(available_after=tick, axis=axis, sign=sign, loss=loss, margin=gap,
                                  supported=gap>=MIN_GAP and loss<=MAX_LOSS, candidates=candidates))
            self.order.append(axis)
            self.signs.append(sign)
            self.seconds += perf_counter()-clock
        self.start_image = image

    def result(self):
        if len(self.fits) == WIDTH and all(row['supported'] for row in self.fits):
            return Wiring(tuple(self.order), tuple(self.signs))
        return None

    def receipt(self):
        return dict(fits=self.fits, wrist_rgb=self.rgb, model_seed=MODEL_SEED, model_trials=self.trials,
                    model_frames=self.model_frames, fit_wall_seconds=self.seconds)


class CoursePlayer:
    def __init__(self, source, wiring, mode, prior_rows, acquisition):
        correction = None if acquisition is None else acquisition['correction']
        correction = None if correction is None else Wiring.restore(correction)
        self.station, self.far, self.bridge = native_station(source, source['profile'], wiring, correction)
        self.fit = WristFit() if mode == 'online' and not prior_rows else None
        self.probing = False
        self.ready = None if acquisition is None else acquisition['acquired_at']
        self.public_ticks = sum(len(row['actions']) for row in prior_rows)
        self.score = prior_rows[-1]['score_after'].copy() if prior_rows else [0, 0]
        self.base_command = self.station.player.command
        self.station.player.command = self.command

    def command(self, tick, position, *, teacher=None):
        if self.probing and tick <= TAP_FRAMES:
            player = self.station.player
            if self.fit is not None and tick % TAP_TICKS == 0:
                self.fit.observe(tick, self.station.arena.render_wrist())
            if tick < TAP_FRAMES:
                player.history.append((tick, position))
                return public_probe(tick)
            correction = None if self.fit is None else self.fit.result()
            self.bridge.correction = correction
            if correction is not None:
                self.ready = self.public_ticks + tick
            for seen in map(public_probe, range(TAP_FRAMES)):
                action = seen if correction is None else correction.physical(seen)
                player.sim.twin.advance(action)
        return self.base_command(tick, position, teacher=teacher)

    def point(self, seed, first):
        self.probing = first
        self.bridge.clear()
        self.far.steps.clear()
        row = self.station.point(seed)
        public, physical, actions = self.bridge.public.copy(), self.bridge.realized.copy(), row['actions']
        ticks = len(actions)
        before = self.score.copy()
        if row['reward']:
            self.score[int(row['reward'] < 0)] += 1
        correction = self.bridge.correction
        row.update(issued_public=public, realized_near=physical, rival_actions=self.far.tape(ticks),
                   actuation=recode(actions, public, physical), public_sha256=digest_json(public),
                   point_starts_at=self.public_ticks, probe_ticks=min(ticks, TAP_FRAMES) if first else 0,
                   active_from=(TAP_FRAMES if first else 0) if correction is not None else None,
                   correction=None if correction is None else correction.state(),
                   score_before=before, score_after=self.score.copy())
        self.public_ticks += ticks
        return row

    def close(self):
        self.station.close()
        if self.fit is not None:
            self.fit.close()


def course_seeds(source):
    known = source['rows']['frozen']
    seeds = [row['seed'] for row in known]
    if seeds != np.random.default_rng(source['seed']).integers(2**31, size=12).tolist():
        raise ValueError('expected the predeclared five-axis course in order')
    if set(seeds) & (set(source['training_seeds']) | {MODEL_SEED}):
        raise ValueError('point seeds overlap model or partner practice')
    return seeds


def evaluate(path, *, resume=False):
    source = _read(SOURCE)
    seeds = course_seeds(source)
    wiring = installation(INSTALLATION)
    data = dict(schema='tt22-live-body-course-1', cohort_sha256=_hash(SOURCE), cohort_seed=source['seed'],
                seeds=seeds, installation_seed=INSTALLATION, evaluator_wiring=wiring.state(),
                profile=source['profile'], opening=dict(spread=SPREAD, pace=PACE), arena=CONFIG,
                probe_ticks=TAP_FRAMES, probe_rate=TAP_RATE, confidence=dict(min_margin=MIN_GAP, max_loss=MAX_LOSS),
                evaluation=dict(game_type='receiver point stream', probes_share_first_scored_point=True,
                                extra_target_resets=0, transfer_eye='wrist', rally_eyes=['left','right']),
                rows={name: [] for name in MODES})
    if resume:
        saved = _read(path)
        if any(digest_json(saved[key]) != digest_json(value) for key, value in data.items() if key != 'rows'):
            raise ValueError('the checkpoint belongs to a different course')
        collected = {(index, name) for name, rows in saved['rows'].items() for index in range(len(rows))}
        schedule = [(index, name) for index in range(len(seeds))
                    for name in (MODES if index % 2 == 0 else MODES[::-1])]
        if collected != set(schedule[:len(collected)]) or any(
                seeds[:len(rows)] != [row['seed'] for row in rows] for rows in saved['rows'].values()):
            raise ValueError('the checkpoint does not contain an uninterrupted prefix')
        if saved['rows']['online'] and 'acquisition' not in saved:
            raise ValueError('the checkpoint has no recorded wrist acquisition')
        data = saved
    players = {name: CoursePlayer(source, wiring, name, data['rows'][name],
                                  data.get('acquisition') if name == 'online' else None) for name in MODES}
    try:
        for i, seed in enumerate(data['seeds']):
            order = MODES if i % 2 == 0 else MODES[::-1]
            for name in order:
                if i < len(data['rows'][name]):
                    continue
                player = players[name]
                row = player.point(seed, i == 0)
                data['rows'][name].append(row)
                if i == 0 and name == 'online':
                    data['acquisition'] = dict(player.fit.receipt(), acquired_at=player.ready,
                                               correction=row['correction'], learned_exact=row['correction']==wiring.state())
                    print('acquisition', data['acquisition']['acquired_at'], data['acquisition']['learned_exact'],
                          [(r['axis'],r['sign'],round(r['margin'],3),round(r['loss'],3))
                           for r in player.fit.fits], flush=True)
                _write(path, data)
                print(name, i+1, seed, row['score_after'], row['longest'], row['outcome']['legal_returns'], flush=True)
    finally:
        for player in players.values():
            player.close()
    data['summary'] = {name: summary(rows) for name, rows in data['rows'].items()}
    data['replayed'] = verify(data)
    _write(path, data)
    print(data['summary'], data['replayed'], flush=True)


def verify(data):
    seeds = course_seeds(_read(SOURCE))
    if (data['cohort_sha256'] != _hash(SOURCE) or data['seeds'] != seeds
            or set(data['rows']) != set(MODES) or any(len(rows) != len(seeds) for rows in data['rows'].values())):
        raise ValueError('the replay needs all 12 matched launches from the original course')
    wiring = installation(INSTALLATION)
    if data['installation_seed'] != INSTALLATION or data['evaluator_wiring'] != wiring.state():
        raise ValueError('the physical installation changed')
    acquired = data['acquisition']
    expected_wrist = {int(tick): digest for tick, digest in acquired['wrist_rgb'].items()}
    if sorted(expected_wrist) != list(range(0, TAP_FRAMES + 1, TAP_TICKS)):
        raise ValueError('the wrist receipt must contain every scheduled observation')
    arena = JitteredOpening(**CONFIG)
    fit = WristFit()
    result = {}
    try:
        for name, rows in data['rows'].items():
            score = [0, 0]
            clock = 0
            for i, row in enumerate(rows):
                ticks = len(row['actions'])
                first = i == 0
                arena.reset(seed=row['seed'], server=-1, render=False)
                correction = None if row['correction'] is None else Wiring.restore(row['correction'])
                if (row['point_starts_at'] != clock or row['score_before'] != score
                        or row['seed'] != seeds[i] or row['role'] != -1):
                    raise ValueError('the stream skips a point, time, or score')
                expected_correction = acquired['correction'] if name == 'online' else None
                active_from = (TAP_FRAMES if first else 0) if expected_correction is not None else None
                if (row['correction'] != expected_correction or row['active_from'] != active_from
                        or row['probe_ticks'] != (TAP_FRAMES if first else 0)):
                    raise ValueError('the correction or common probe time changed')
                digest = sha256()
                for tick, (nominal, public, physical, far) in enumerate(zip(row['actions'], row['issued_public'],
                        row['realized_near'], row['rival_actions'], strict=True)):
                    active = correction is not None and tick >= row['active_from']
                    translated = correction.inverse(nominal).tolist() if active else nominal
                    if translated != public or wiring.physical(public).tolist() != physical:
                        raise ValueError('the replayed public command changed')
                    if tick < row['probe_ticks'] and public_probe(tick).tolist() != public:
                        raise ValueError('the scorer did not run the predeclared common probe')
                    if first and tick in expected_wrist:
                        image = arena.render_wrist()
                        if sha256(image).hexdigest() != expected_wrist[tick]:
                            raise ValueError('the recorded wrist evidence did not replay')
                        if name == 'online':
                            fit.observe(tick, image)
                    for eye, frame in arena.render_cameras().items():
                        digest.update(eye.encode())
                        digest.update(frame)
                    arena._rival_attempt = True
                    arena._aim = lambda *args: np.asarray(far)
                    _, reward, done, cap, _ = arena.step(wiring.physical(public), render=False)
                    if bool(done or cap) != (tick + 1 == ticks):
                        raise ValueError('the point stopped on a different tick')
                if reward:
                    score[int(reward < 0)] += 1
                if (reward != row['reward'] or row['score_after'] != score or arena.point_result() != row['outcome']
                        or digest_json(arena.events) != digest_json(row['events'])
                        or digest.hexdigest() != row['rgb'] or alternating(arena.events) != row['longest']):
                    raise ValueError(f'the referee, score or RGB sequence changed for {name} seed {row["seed"]}')
                if (digest_json(row['issued_public']) != row['public_sha256']
                        or recode(row['actions'], row['issued_public'], row['realized_near']) != row['actuation']):
                    raise ValueError('the public command receipt does not match its tapes')
                clock += ticks
            if 'summary' in data and summary(rows) != data['summary'][name]:
                raise ValueError('the point summary disagrees with the replay')
            result[name] = dict(physics=len(rows), stereo=len(rows), final_score=score)
        inferred = fit.result()
        if (acquired['correction'] != (None if inferred is None else inferred.state())
                or acquired['learned_exact'] != (inferred == wiring)
                or acquired['acquired_at'] != (None if inferred is None else TAP_FRAMES)
                or any(digest_json(value) != digest_json(acquired[key]) for key, value in fit.receipt().items()
                       if key != 'fit_wall_seconds')):
            raise ValueError('independent fitting from the replayed wrist RGB changed the acquisition')
    finally:
        arena.close()
        fit.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT/'tt22_live_body.json.gz')
    parser.add_argument('--phase', choices=('course', 'verify'), default='course')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'course':
        evaluate(args.output, resume=args.resume)
    else:
        print(verify(_read(args.output)))
