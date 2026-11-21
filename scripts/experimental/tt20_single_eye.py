"""Learn an opening stroke from one RGB camera and public point rewards."""
from copy import deepcopy
import numpy as np

from .tt18_view import ImageColors, OBJECT_HUES
from .tt19_view import _blob
from .tt4_transfer_trial import _read

# The opening ball is observed before moving. These are control frames at 30 Hz.
SAMPLES = {-1: (0, 4, 8), 1: (0, 1, 2)}
TIMING = np.array([-2., 0., 2.])


def image_ball(rgb):
    colors = ImageColors(rgb)
    ball = _blob(colors, OBJECT_HUES['ball'], tolerance=.034, minimum=62., area=1,
                 span=max(9, rgb.shape[0] // 8))
    if ball is None:
        return None
    return [float(ball.center[0]/rgb.shape[1]), float(ball.center[1]/rgb.shape[0]), len(ball.pixels)]


def cue(samples):
    seen = [(tick, np.asarray(value)) for tick, value in samples if value is not None]
    if not seen:
        return None
    t, p = seen[-1]
    first, a = seen[0]
    slope = (p - a) / max(1, t-first)
    return [float(p[0] + 4*slope[0]), float(p[1] + 3*slope[1])]


def unique_programs(rows, count):
    rows = np.asarray(rows)
    selected = [0]
    # Cover the previous body's repertoire without using its scores for selection.
    scale = np.maximum(np.std(rows, axis=0), .18)
    while len(selected) < count:
        d = ((rows[:, None] - rows[selected]) / scale)**2
        j = int(np.argmax(d.mean(axis=2).min(axis=1)))
        if j in selected:
            break
        selected.append(j)
    return rows[selected]


def motor_library(folder, role):
    bank = _read(folder/'consolidated-full-from-wide-bank.json.gz')['banks'][str(role)]['programs']
    initial = unique_programs(bank, 5)
    if role == 1:
        factory = _read(folder/'factory-serve-proposals.json.gz')['programs']
        initial = np.vstack([factory[0], initial, factory[1]])
    return initial


class Memory:
    """Separate sensor centering from motor values that can cross between views."""

    def __init__(self, programs, state=None):
        self.programs = np.asarray(programs)
        self.n = len(programs)*len(TIMING)
        self.rows = []
        self.eyes = {}
        if state:
            self.rows = deepcopy(state['rows'])
            self.eyes = deepcopy(state['eyes'])

    def state(self):
        return dict(rows=deepcopy(self.rows), eyes=deepcopy(self.eyes))

    def features(self, camera, seen, *, remember=False):
        history = self.eyes.get(camera, [])
        if seen is None:
            return np.zeros(2)
        if len(history) >= 6:
            a = np.asarray(history[-96:])
            center = np.median(a, axis=0)
            scale = np.maximum(.5*(np.quantile(a,.84,axis=0)-np.quantile(a,.16,axis=0)), [.006,.006])
            relative = np.clip((np.asarray(seen)-center)/scale, -3, 3)
        else:
            relative = np.zeros(2)
        if remember:
            self.eyes.setdefault(camera, []).append(seen)
        return relative

    def choose(self, x, *, explore=False, rng=None):
        # A neutral prior assigns equal evidence to every stroke/timing pair.
        if not self.rows:
            return int(rng.integers(self.n)) if explore else 1
        rows = np.asarray(self.rows)
        actions = rows[:,2].astype(int)
        delta = rows[:,:2]-x
        kernel = .20 + .80*np.exp(-.5*((delta/[.85,1.35])**2).sum(axis=1))
        count = np.bincount(actions, weights=kernel, minlength=self.n)
        wins = np.bincount(actions, weights=kernel*rows[:,3], minlength=self.n)
        mean = (.8+wins)/(2+count)
        if explore:
            # Thompson sampling spends early practice on distinct choices.
            mean = rng.beta(wins+.8, count-wins+1.2)
        return int(np.argmax(mean))

    def record(self, x, action, reward):
        self.rows.append([float(x[0]),float(x[1]),int(action),float(reward>0)])

    def commands(self, index, x, role, max_steps):
        program = self.programs[index//len(TIMING)]
        base, swing = program[:5].copy(), program[5:10].copy()
        # The stable home views have the same image handedness. The scale of
        # lateral variation comes from the learner's own image stream.
        base[0] += .95 * np.clip(x[0], -2.2, 2.2)
        onset = int(round(program[10] + TIMING[index%len(TIMING)]))
        width = max(2, round(program[11]))
        t = np.arange(max_steps)
        age = t-onset
        gain = np.where(age>=0,np.clip(np.minimum((age+1)/width, (2*width+3-age)/width),0,1),0)
        goals = base + gain[:,None]*swing
        goals[:SAMPLES[role][-1]] = 0
        goals[t>onset+2*width+6] = 0
        positions = np.zeros(5)
        actions = np.zeros((max_steps,6))
        for i, goal in enumerate(goals):
            actions[i,:5] = np.clip(goal-positions,-.85,.85)
            positions += actions[i,:5]
        return actions
