"""An RGB falling-fruit game for testing transfer of visual motor timing."""

import numpy as np

from .tt6_timing import TimingMemory, VisualDelay
from .tt7_transfer_timing import ChangingDelay, DynamicsMemory


WIDTH, HEIGHT = 144, 88
LINE = 76
GRAVITY = .18
RATE = 4.4
RADIUS = 5.5
X_MIN, X_MAX = 7., WIDTH - 8.
BASKET = (76, 208, 249)
FRUIT = (255, 174, 50)


class FruitCourt:
    """Slide a basket along a rail to catch falling fruit in a rendered court."""

    action_size = 1

    def __init__(self, *, occlusion=False):
        self.arena = self
        self.occlusion = occlusion
        self.backdrop = np.full((HEIGHT, WIDTH, 3), (22, 32, 51), dtype=np.uint8)
        self.backdrop[LINE + 3:] = (35, 54, 69)
        self.backdrop[LINE + 3, :] = (82, 108, 119)
        self.backdrop[11:LINE:8, 4:WIDTH - 4:9] = (32, 45, 65)

    def reset(self, *, seed):
        rng = np.random.default_rng(seed)
        self.step_count = 0
        self.basket = float(rng.uniform(51., 93.))
        self.flight = int(rng.integers(16, 29))
        self.wind_tick = int(rng.integers(self.flight // 3, 2 * self.flight // 3))
        self.x_start = rng.uniform(35., 109.)
        self.x_velocity = rng.uniform(-3.8, 3.8)
        self.wind = rng.uniform(-5.5, 5.5)
        self.x, self.y = self.x_start, 7.
        self.y_velocity = (LINE - self.y) / self.flight - .5 * GRAVITY * self.flight
        return self.frame()

    def frame(self):
        rgb = self.backdrop.copy()
        x, y = round(self.x), round(self.y)
        rgb[y - 1:y + 2, x - 2:x + 3] = FRUIT
        rgb[y - 2:y + 3, x - 1:x + 2] = FRUIT
        if self.occlusion and y >= 46:
            rgb[LINE:LINE + 4, :, :] = (50, 71, 84)
        else:
            x = round(self.basket)
            rgb[LINE + 1:LINE + 3, x - 5:x + 6] = BASKET
            rgb[LINE - 1:LINE + 1, [x - 5, x + 5]] = BASKET
        return rgb

    def step(self, action, *, render=True):
        self.step_count += 1
        t = self.step_count
        self.basket = float(np.clip(self.basket + RATE * action[0], X_MIN, X_MAX))
        horizontal = self.x_start + t * self.x_velocity + max(0, t - self.wind_tick) * self.wind
        # Reflect the flight at the court walls without hiding either object.
        width = X_MAX - X_MIN
        self.x = float(X_MIN + width - abs((horizontal - X_MIN) % (2 * width) - width))
        self.y = 7. + self.y_velocity * t + .5 * GRAVITY * t**2
        done = t == self.flight
        caught = done and abs(self.x - self.basket) <= RADIUS
        return self.frame() if render else None, float(caught), done, False, {}


def gaze(rgb):
    """Read object centers from pixels, with the court colors public."""
    # The colors have distinct red channels. The extra cart markers keep its
    # centroid visible through contact, even if the fruit overlays the rail.
    line = rgb[:, :, 0]
    fruit_y, fruit_x = np.nonzero(line == FRUIT[0])
    _, basket_x = np.nonzero(line[LINE - 1:LINE + 3] == BASKET[0])
    basket = float(basket_x.mean()) if basket_x.size else None
    return float(fruit_x.mean()), float(fruit_y.mean()), basket


class CatchPlayer:
    """Lead a moving fruit by the transport lag inferred from basket motion."""

    def __init__(self, donor=None, *, learn=True, dynamics=False):
        self.memory = DynamicsMemory(donor) if dynamics else TimingMemory(donor)
        self.learn = learn
        self.dynamics = dynamics

    def reset(self, rgb):
        _, _, basket = gaze(rgb)
        self.sightings = []
        self.nominal = basket
        if self.dynamics:
            self.observer = ChangingDelay(self.memory, [basket], [.85])
        else:
            self.observer = VisualDelay(self.memory, [basket], [.85], learn=self.learn)

    def act(self, rgb, preceding=None):
        x, y, basket = gaze(rgb)
        if preceding is not None:
            self.nominal = float(np.clip(self.nominal + RATE * preceding[0], X_MIN, X_MAX))
            self.observer.advance([self.nominal])
        self.observer.observe([basket] if basket is not None else None)
        self.sightings.append((x, y))
        t = min(3, len(self.sightings) - 1)
        if t:
            old_x, old_y = self.sightings[-1 - t]
            vx, vy = (x - old_x) / t, (y - old_y) / t + .5 * GRAVITY * t
            remaining = (np.sqrt(vy**2 + 2 * GRAVITY * (LINE - y)) - vy) / GRAVITY
        else:
            vx, remaining = 0., np.inf
        lead = min(remaining, 1. + self.observer.delay)
        width = X_MAX - X_MIN
        projected = x + vx * lead
        target = X_MIN + width - abs((projected - X_MIN) % (2 * width) - width)
        # Correct short drift without cancelling the visual lead. Every group
        # receives the same tracker, motor units, and interception policy.
        predicted = float((self.observer.belief @ np.asarray(self.observer.history))[0])
        error = 0. if basket is None else basket - predicted
        command = (target - self.nominal - .35 * error) / RATE
        return np.array([np.clip(command, -1., 1.)])

    def finish(self):
        self.observer.finish()
