"""An optional table course with preparation time and a full two-bounce serve."""

import math
import random

try:
    from .tt3_arena import (BALL_RADIUS, GRAVITY, HZ, NET_HEIGHT, TABLE_HALF_LENGTH,
                            TABLE_HALF_WIDTH, TABLE_RESTITUTION, TT3Arena)
except ImportError:
    from tt3_arena import (BALL_RADIUS, GRAVITY, HZ, NET_HEIGHT, TABLE_HALF_LENGTH,
                           TABLE_HALF_WIDTH, TABLE_RESTITUTION, TT3Arena)


def _full_serve(rng):
    """Work backward from a receiver landing to a legal bounce in each half."""
    while True:
        hit_x = rng.choice((-1., 1.)) * rng.uniform(.16, .49)
        hit_y, hit_z = rng.uniform(-1.45, -1.17), rng.uniform(.19, .39)
        pace = rng.uniform(3.12, 3.74)
        after_receiver_bounce = rng.uniform(.16, .26)
        receiver_depth = hit_y + pace * after_receiver_bounce
        receiver_up = ((hit_z - BALL_RADIUS) / after_receiver_bounce
                       + GRAVITY * after_receiver_bounce / 2)
        server_up = receiver_up / TABLE_RESTITUTION
        airborne = 2 * server_up / GRAVITY
        server_depth = receiver_depth + pace * airborne
        if not (-1.28 < receiver_depth < -.39 and .50 < server_depth < 1.27):
            continue
        from_bounce_to_net = server_depth / pace
        above_net = BALL_RADIUS + from_bounce_to_net * (server_up - GRAVITY * from_bounce_to_net / 2)
        if above_net < NET_HEIGHT + BALL_RADIUS + .026:
            continue
        start_y = rng.uniform(1.35, 1.55)
        before_bounce = (start_y - server_depth) / pace
        start_vz = GRAVITY * before_bounce - server_up / TABLE_RESTITUTION
        start_z = BALL_RADIUS - start_vz * before_bounce + GRAVITY * before_bounce**2 / 2
        if not .16 < start_z < .55:
            continue
        start_x = .53 * hit_x + rng.uniform(-.26, .26)
        flight = before_bounce + airborne + after_receiver_bounce
        return (start_x, start_y, start_z), ((hit_x - start_x) / flight, -pace, start_vz)


class PreparedTT3Arena(TT3Arena):
    """Show a pre-serve cue, then play ordinary points after a legal two-bounce serve."""

    def __init__(self, seed=0, *, view_seed=None, preparation_frames=24,
                 max_frames=None, distractor=False, variant='standard'):
        if variant != 'standard':
            raise ValueError('full-table preparation is available on the standard course')
        if type(preparation_frames) is not int or not 1 <= preparation_frames <= 60:
            raise ValueError('preparation time must be between one and 60 frames')
        if max_frames is None:
            max_frames = 300 + preparation_frames
        if type(max_frames) is not int or max_frames <= preparation_frames:
            raise ValueError('frame budget must extend beyond the preparation')
        self.preparation_frames = preparation_frames
        super().__init__(seed, view_seed=view_seed, max_frames=max_frames, distractor=distractor)

    def reset(self, *, render=True):
        super().reset(render=False)
        rng = random.Random(f'tt3/full-serve/v1/{self.seed}/{self._point}')
        self._launch, self._launch_velocity = _full_serve(rng)
        self._server_bounce_due = False
        self._prepare()
        return self.frame() if render else None

    def _prepare(self):
        fraction = self.frames / self.preparation_frames
        time = self.preparation_frames / HZ
        x, y, z = self._launch
        vx, _, _ = self._launch_velocity
        # Carry the ball forward before the strike so its approach direction
        # and the requested arm motion are both visible during preparation.
        self.ball = (x - .085 * vx * (1 - fraction), y + .22 * (1 - fraction),
                     z + .058 * math.sin(math.pi * fraction))
        self.velocity = (.085 * vx / time, -.22 / time,
                         .058 * math.pi * math.cos(math.pi * fraction) / time)
        self.rival = (x + .16, y + .19, max(.09, z - .12))

    def step(self, action, *, render=True):
        preparing = self.frames < self.preparation_frames
        _, reward, done, info = super().step(action, render=False)
        if preparing:
            if self.frames < self.preparation_frames:
                self._prepare()
            else:
                self.ball, self.velocity = self._launch, self._launch_velocity
                self._server_bounce_due = True
                launch = dict(kind='serve', frame=self.frames, side=-1, ball=self.ball,
                              velocity=self.velocity)
                self.events.append(launch)
                info['events'] = (*info['events'], launch)
        return self.frame() if render else None, reward, done, info

    def _table_and_bounds(self, old_ball, dt):
        x, y, z = self.ball
        vx, vy, vz = self.velocity
        if self._server_bounce_due and old_ball[2] >= BALL_RADIUS > z and vz < 0.:
            prior_vz = vz + GRAVITY * dt
            elapsed = (prior_vz + math.sqrt(prior_vz**2 + 2 * GRAVITY * (old_ball[2] - BALL_RADIUS))) / GRAVITY
            bx, by = old_ball[0] + vx * elapsed, old_ball[1] + vy * elapsed
            if abs(bx) <= TABLE_HALF_WIDTH and 0. < by <= TABLE_HALF_LENGTH:
                self._event('serve_bounce', side=-1, location=(bx, by, BALL_RADIUS))
                remaining = dt - elapsed
                upward = TABLE_RESTITUTION * (GRAVITY * elapsed - prior_vz)
                self.ball = (x, y, BALL_RADIUS + upward * remaining - GRAVITY * remaining**2 / 2)
                self.velocity = (vx, vy, upward - GRAVITY * remaining)
                self._server_bounce_due = False
                return
        super()._table_and_bounds(old_ball, dt)
