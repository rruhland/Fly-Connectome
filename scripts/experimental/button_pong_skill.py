"""Practice native visual Pong independently of the arm and button implementation."""

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from pong_score_placement import OFFSETS, ScorePlacement


CAMERA_SIZE = 192
CAMERA_TICKS = 4
TICKS_PER_SECOND = 120
CONTACT_X = .0625
MEASUREMENT_X = .125
SCORING_OPPONENT_SPEED = .35
PRACTICE_EXPLORATION = .42
LEAD_SCALES = (0., .25, .5, .75, .9, .95, 1., 1.05, 1.1, 1.25, 1.5)
SOURCE_SEED_MIN = 100_000_000
SOURCE_SEED_MAX = 600_000_000
STATE_SCHEMA = 'button-pong-visual-strategy-v1'


@dataclass(frozen=True)
class PongScreen:
    """Positions decoded from the native raster, in down-positive screen units."""

    paddle_y: float
    opponent_y: float
    ball: tuple[float, float] | None

    @classmethod
    def read(cls, frame):
        field = np.asarray(frame)
        if field.ndim != 2 or field.dtype != np.bool_ or min(field.shape) < 64:
            raise ValueError('a two-dimensional native binary Pong frame is required')
        height, width = field.shape
        paddle_column = int(.05 * width)
        left = np.flatnonzero(field[:, paddle_column])
        right = np.flatnonzero(field[:, width - 1 - paddle_column])
        if not len(left) or not len(right):
            raise ValueError('both paddles must be visible')

        def center(rows):
            # The paddle occupies many consecutive rows. A nearby ball can add
            # a few pixels at either end without corrupting the median.
            return float((np.median(rows) + .5) / height)

        first = math.floor(CONTACT_X * width - .5) + 1
        last = width - first
        inside = field[:, first:last]
        columns = np.flatnonzero(inside.any(axis=0))
        ball = None
        if len(columns):
            rows = np.flatnonzero(inside[:, columns[0]:columns[-1] + 1].any(axis=1))
            ball = (float((columns[0] + columns[-1] + 1.) / (2 * width) + first / width),
                    float((rows[0] + rows[-1] + 1.) / (2 * height)))
        return cls(center(left), center(right), ball)


def reflected_y(unfolded):
    """Fold the projected vertical flight into the visible screen boundaries."""
    low = .0125
    width = 1. - 2 * low
    return low + width - abs((unfolded - low) % (2 * width) - width)


def projection(position, velocity, crossing_x, lead):
    x, y = position
    vx, vy = velocity
    return reflected_y(y + lead * vy * ((crossing_x - x) / vx))


class CrossingEvidence:
    """Learn the amount to lead from ball positions before a visible crossing."""

    def __init__(self, losses=None, count=0):
        self.losses = np.zeros(len(LEAD_SCALES)) if losses is None else np.array(losses, dtype=float)
        if (self.losses.shape != (len(LEAD_SCALES),) or not np.isfinite(self.losses).all()
                or np.any(self.losses < 0) or type(count) is not int or count < 0
                or (count == 0 and np.any(self.losses))):
            raise ValueError('invalid visible-crossing evidence')
        self.count = count
        self.lead = 0.
        self._select()

    def _select(self):
        if self.count:
            # Initial preference for the observed rather than extrapolated y.
            # Enough image-measured crossings must outweigh this finite prior.
            scales = np.asarray(LEAD_SCALES)
            self.lead = LEAD_SCALES[int(np.argmin(self.losses + .035 * scales**2))]

    def observe(self, glimpses, y):
        estimates = [[projection(pos, vel, MEASUREMENT_X, scale) for scale in LEAD_SCALES]
                     for pos, vel in glimpses]
        self.losses += np.square(np.asarray(estimates) - y).mean(axis=0)
        self.count += 1
        self._select()

    def to_state(self):
        return dict(losses=self.losses.tolist(), count=self.count)


class DesiredPongPolicy:
    """Camera-to-game intent with learned interception and terminal score placement."""

    def __init__(self, *, crossings=None, scores=None, learn=True, exploration=.10, aim=True):
        if not math.isfinite(exploration) or not 0. <= exploration <= 1.:
            raise ValueError('exploration probability must be in [0, 1]')
        self.crossings = CrossingEvidence() if crossings is None else crossings
        self.scores = ScorePlacement() if scores is None else scores
        self.learn = learn
        self.exploration = exploration
        self.aim = aim
        self.active = False

    @classmethod
    def load(cls, path, *, learn=True, exploration=.10, aim=True):
        state = json.loads(Path(path).read_text())
        if state['schema'] != STATE_SCHEMA:
            raise ValueError('unknown visual Pong strategy checkpoint')
        return cls(crossings=CrossingEvidence(**state['crossings']),
                   scores=ScorePlacement(**state['scores']), learn=learn,
                   exploration=exploration, aim=aim)

    def to_state(self):
        return dict(schema=STATE_SCHEMA, crossings=self.crossings.to_state(),
                    scores=self.scores.to_state())

    def begin(self, seed):
        """Start a point from a fresh public track, retaining prior learned evidence."""
        if self.active:
            raise ValueError('finish the previous Pong point before starting another')
        self.active = True
        self.rng = np.random.default_rng(seed)
        self.tick = None
        self._paddle = deque(maxlen=3)
        self._ball = deque(maxlen=7)
        self._glimpses = {}
        self._choice = None
        self._last_scored_choice = None
        self.paddle_velocity = 0.
        self.ball_velocity = None
        self._target = .5
        self.crossing_errors = []
        self.choices = []

    def _track(self, ball, now):
        if ball is None:
            if self._ball and now - self._ball[-1][0] > .15:
                self._ball.clear()
                self.ball_velocity = None
            return None
        x, y = ball
        previous = self._ball[-1] if self._ball else None
        velocity = self.ball_velocity
        if previous and velocity:
            _, old_x, old_y = previous
            vx, vy = velocity
            reversed_x = (x - old_x) * vx < -.002
            wall_turn = min(old_y, 1. - old_y) < .065 and (y - old_y) * vy < -.00065
            if reversed_x or wall_turn:
                self._ball.clear()
                if wall_turn and not reversed_x:
                    self._ball.append(previous)
                self.ball_velocity = None
            if reversed_x:
                self._glimpses.clear()
                self._choice = None
        self._ball.append((now, x, y))
        if len(self._ball) >= 2:
            t, x0, y0 = self._ball[0]
            elapsed = now - t
            self.ball_velocity = ((x - x0) / elapsed, (y - y0) / elapsed)
        return previous

    def _observe_crossing(self, ball, previous):
        x, y = ball
        velocity = self.ball_velocity
        for line in (.70, .45, .28):
            if previous[1] > line >= x and velocity[0] < -.15:
                self._glimpses[line] = (ball, velocity)
        if previous[1] > MEASUREMENT_X >= x and self._glimpses:
            portion = (previous[1] - MEASUREMENT_X) / (previous[1] - x)
            measured = previous[2] + portion * (y - previous[2])
            examples = tuple(self._glimpses.values())
            forecasts = [projection(position, speed, MEASUREMENT_X, self.crossings.lead)
                         for position, speed in examples]
            self.crossing_errors.extend(abs(value - measured) for value in forecasts)
            if self.learn:
                self.crossings.observe(examples, measured)
            self._glimpses.clear()

    def _aim(self, intercept, remaining, paddle):
        if not self.aim or self.crossings.lead < .75:
            return intercept
        sign = 1. if intercept >= .5 else -1.
        if self._choice is None and .40 <= remaining <= 1.05:
            max_travel = .64 * (remaining - .13)
            reachable = (max(.10, paddle - max_travel), min(.90, paddle + max_travel))
            legal = [0]
            legal.extend(index for index, offset in enumerate(OFFSETS[1:], 1)
                         if reachable[0] <= intercept + sign * offset <= reachable[1])
            if len(legal) > 1:
                arm = self.scores.choose(intercept - .5, legal, self.rng, self.exploration)
                self._choice = (arm, intercept, sign)
                self._last_scored_choice = self._choice
                self.choices.append((arm, intercept))
        if self._choice is None:
            return intercept
        arm, _, committed_sign = self._choice
        return intercept + OFFSETS[arm] * committed_sign

    def target(self, screen, tick):
        """Return desired paddle y from a native public screen sample."""
        if not self.active or (self.tick is not None and tick < self.tick):
            raise ValueError('start a point and use nondecreasing camera ticks')
        if tick == self.tick:
            return self._target
        self.tick = tick
        now = tick / TICKS_PER_SECOND
        self._paddle.append((now, screen.paddle_y))
        if len(self._paddle) >= 2:
            then, old = self._paddle[0]
            self.paddle_velocity = (screen.paddle_y - old) / (now - then)
        ball = screen.ball
        previous = self._track(ball, now)
        velocity = self.ball_velocity
        if ball is not None and velocity is not None and previous is not None:
            self._observe_crossing(ball, previous)
        if ball is None:
            return self._target
        x, y = ball
        if velocity is None or velocity[0] >= -.15:
            self._target = y if velocity is None or x > .70 else .5
        else:
            remaining = max(0., (CONTACT_X - x) / velocity[0])
            intercept = reflected_y(y + self.crossings.lead * velocity[1] * remaining)
            self._target = self._aim(intercept, remaining, screen.paddle_y)
        self._target = min(.90, max(.10, self._target))
        return self._target

    def action(self, screen, tick, *, stop_seconds=.30):
        """Return requested game direction: -1 up, +1 down, and 0 release."""
        target = self.target(screen, tick)
        error = target - (screen.paddle_y + stop_seconds * self.paddle_velocity)
        return int(error > .012) - int(error < -.012)

    def finish(self, score):
        """Update the last shot choice after a native point or declared timeout."""
        if not self.active or score not in (-1, 0, 1):
            raise ValueError('one native score or timeout is required per point')
        if self.learn and self._last_scored_choice is not None:
            arm, intercept, _ = self._last_scored_choice
            self.scores.observe_score(arm, intercept - .5, score)
        self.active = False


class NaivePongPolicy(DesiredPongPolicy):
    """Use the matched reactive chase and retain no outcome evidence."""

    def __init__(self):
        super().__init__(learn=False, exploration=0., aim=False)


def native_points(seeds, policies, *, opponent_speed=SCORING_OPPONENT_SPEED, max_ticks=1800):
    """Run native physics and keep its private collision data in the evaluator."""
    import torch

    from fly_connectome.pong import Physics, Pong

    if len(seeds) != len(policies):
        raise ValueError('one policy per native starting seed is required')
    game = Pong(seeds, Physics(opponent_speed=opponent_speed))
    drive = torch.zeros(len(seeds))
    active = torch.ones(len(seeds), dtype=torch.bool)
    opportunities = torch.zeros(len(seeds), dtype=torch.int64)
    hits = torch.zeros_like(opportunities)
    first_hits = torch.zeros_like(opportunities)
    rows = [None] * len(seeds)
    for seed, policy in zip(seeds, policies, strict=True):
        policy.begin(seed)
    for tick in range(max_ticks):
        if tick % CAMERA_TICKS == 0:
            indices = active.nonzero().flatten().tolist()
            frames = game.render(CAMERA_SIZE, CAMERA_SIZE).numpy()
            for index in indices:
                screen = PongScreen.read(frames[index])
                drive[index] = -policies[index].action(screen, tick)
        x = game.ball[:, 0]
        vx = game.ball_velocity[:, 0]
        own = (x >= CONTACT_X) & (x + vx * game.config.dt <= CONTACT_X) & (vx < 0) & active
        outcome = game.step(drive, auto_reset=False)
        returned = outcome.hits.to(torch.int64)
        first_hits += ((opportunities == 0) & own) * returned
        opportunities += own
        hits += returned * active
        finished = active & ((outcome.scores != 0) | (tick == max_ticks - 1))
        for index in finished.nonzero().flatten().tolist():
            score = int(outcome.scores[index])
            policy = policies[index]
            policy.finish(score)
            rows[index] = dict(seed=seeds[index], score=score, own_hit_win=int(score == 1 and hits[index] > 0),
                               hits=int(hits[index]), opportunities=int(opportunities[index]),
                               first_hit=int(first_hits[index]), ticks=tick + 1,
                               camera_crossing_errors=policy.crossing_errors, choices=policy.choices)
        active &= ~finished
        if not active.any():
            break
    return rows


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def summarize(rows):
    first_opportunities = sum(row['opportunities'] > 0 for row in rows)
    errors = [item for row in rows for item in row['camera_crossing_errors']]
    return dict(points=len(rows), native_seconds=sum(row['ticks'] for row in rows) / TICKS_PER_SECOND,
                own_hit_wins=sum(row['own_hit_win'] for row in rows),
                natural_wins=sum(row['score'] == 1 and not row['hits'] for row in rows),
                losses=sum(row['score'] == -1 for row in rows), timeouts=sum(row['score'] == 0 for row in rows),
                returns=sum(row['hits'] for row in rows), opportunities=sum(row['opportunities'] for row in rows),
                first_returns=sum(row['first_hit'] for row in rows), first_opportunities=first_opportunities,
                camera_crossing_estimates=len(errors), mean_absolute_camera_crossing_error=(
                    float(np.mean(errors)) if errors else None))


def paired_interval(learned, initial, numerator, *, denominator=None):
    if [row['seed'] for row in learned] != [row['seed'] for row in initial]:
        raise ValueError('paired evaluation requires identical starts')
    samples = np.random.default_rng(62561811).integers(len(learned), size=(5000, len(learned)))
    distributions = []
    point_estimates = []
    for arm in (learned, initial):
        values = np.asarray([numerator(row) for row in arm])
        exposure = np.ones(len(arm)) if denominator is None else np.asarray([denominator(row) for row in arm])
        sampled_exposure = exposure[samples].sum(axis=1)
        distribution = np.divide(values[samples].sum(axis=1), sampled_exposure,
                                 out=np.full(len(samples), np.nan), where=sampled_exposure != 0)
        distributions.append(distribution)
        point_estimates.append(float(values.sum() / exposure.sum()) if exposure.sum() else None)
    difference = distributions[0] - distributions[1]
    difference = difference[np.isfinite(difference)]
    return dict(difference=(point_estimates[0] - point_estimates[1] if None not in point_estimates else None),
                interval95=np.quantile(difference, [.025, .975]).tolist() if len(difference) else None)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def summarize_source(practice, results, trained):
    crossings, scores = trained.crossings, trained.scores
    return dict(practice={phase: summarize(rows) for phase, rows in practice.items()},
                held_out={condition: summarize(rows) for condition, rows in results.items()},
                crossing_lead=crossings.lead, crossing_training_samples=crossings.count,
                offset_visits=dict(zip(OFFSETS, scores.visits.tolist(), strict=True)),
                trained_minus_cold=paired_interval(results['trained'], results['cold'],
                                                   lambda row: row['own_hit_win']),
                trained_minus_interception=paired_interval(results['trained'], results['trained_interception'],
                                                           lambda row: row['own_hit_win']),
                interception_minus_cold_first_return=paired_interval(
                    results['trained_interception'], results['cold'], lambda row: row['first_hit'],
                    denominator=lambda row: row['opportunities'] > 0),
                interception_minus_cold_all_returns=paired_interval(
                    results['trained_interception'], results['cold'], lambda row: row['hits'],
                    denominator=lambda row: row['opportunities']),
                interception_minus_cold_crossing_error=paired_interval(
                    results['trained_interception'], results['cold'],
                    lambda row: sum(row['camera_crossing_errors']),
                    denominator=lambda row: len(row['camera_crossing_errors'])))


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--interception', type=int, default=96)
    parser.add_argument('--placement', type=int, default=384)
    parser.add_argument('--evaluation', type=int, default=144)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--max-ticks', type=int, default=1800)
    parser.add_argument('--opponent-speed', type=float, default=SCORING_OPPONENT_SPEED)
    parser.add_argument('--seed', type=int, default=3_709_419)
    args = parser.parse_args()
    if min(args.interception, args.placement, args.evaluation, args.batch, args.max_ticks) < 1:
        parser.error('practice, evaluation, batch and tick counts must be positive')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    sizes = dict(interception=args.interception, placement=args.placement, evaluation=args.evaluation)
    all_seeds = rng.choice(SOURCE_SEED_MAX - SOURCE_SEED_MIN, size=sum(sizes.values()), replace=False)
    all_seeds = (all_seeds + SOURCE_SEED_MIN).tolist()
    starts = {}
    position = 0
    for name, size in sizes.items():
        starts[name] = all_seeds[position:position + size]
        position += size
    write_json(args.out / 'source-schedule.json', dict(starts=starts, schedule_seed=args.seed,
               seed_range=[SOURCE_SEED_MIN, SOURCE_SEED_MAX], camera_size=CAMERA_SIZE,
               camera_ticks=CAMERA_TICKS, tick_hz=TICKS_PER_SECOND,
               opponent_speed=args.opponent_speed, max_ticks=args.max_ticks, training_batch_size=args.batch,
               practice_exploration=PRACTICE_EXPLORATION))
    cold = DesiredPongPolicy()
    write_json(args.out / 'source-cold.json', cold.to_state())
    crossings, scores = cold.crossings, cold.scores
    practice = {}
    for phase in ('interception', 'placement'):
        practice[phase] = []
        for block in batches(starts[phase], args.batch):
            policies = [DesiredPongPolicy(crossings=crossings, scores=scores, exploration=PRACTICE_EXPLORATION,
                                          aim=phase == 'placement') for _ in block]
            practice[phase].extend(native_points(block, policies, opponent_speed=args.opponent_speed,
                                                 max_ticks=args.max_ticks))
            print(phase, len(practice[phase]), 'lead', crossings.lead, 'offset visits', scores.visits.tolist(),
                  flush=True)
            write_json(args.out / 'source-practice.json', practice)
        state = cold.to_state()
        write_json(args.out / f'source-{phase}.json', state)
    write_json(args.out / 'source-trained.json', cold.to_state())
    results = {}
    conditions = {
        'cold': (args.out / 'source-cold.json', True),
        'trained_interception': (args.out / 'source-interception.json', False),
        'trained': (args.out / 'source-trained.json', True),
    }
    for condition, (path, aim) in conditions.items():
        results[condition] = []
        for block in batches(starts['evaluation'], args.batch):
            policies = [DesiredPongPolicy.load(path, learn=False, exploration=0., aim=aim) for _ in block]
            results[condition].extend(native_points(block, policies, opponent_speed=args.opponent_speed,
                                                   max_ticks=args.max_ticks))
            print('evaluation', condition, len(results[condition]), summarize(results[condition]), flush=True)
            write_json(args.out / 'source-evaluation.json', results)
    summary = summarize_source(practice, results, cold)
    write_json(args.out / 'source-summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
