"""Separate wrist acquisition failures from physical capture failures in TT24."""

import argparse
from collections import deque
import gzip
import json

import mujoco
import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BALL_RADIUS, BounceArena, WRIST_CAMERA
from .tt24_bounce_ceiling import CeilingSettings, PrivilegedCeiling
from .tt24_bounce_ident import CleanState, _ellipse
from .tt24_broad_donors import fixed_seeds
from .tt24_context_residual import WINDOW, flight
from .tt24_continuous_gravity import GRAVITIES, SEGMENT
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import FUSED_OUTPUT, ROOT, Native, load_public
from .tt24_launch_context import draw
from .tt24_posture import WidePosture


ARCHIVED = ROOT / 'startup-acquisition-archive.json.gz'
HELD = ROOT / 'startup-acquisition-held.json.gz'
REACHABILITY = ROOT / 'startup-acquisition-reachability.json.gz'
FIGURE = ROOT / 'startup-acquisition-vision.png'
LOOK = 64
CLEAN_WINDOW = 12


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def save(path, obj):
    pending = path.with_name(path.name + '.tmp')
    with gzip.open(pending, 'wt') as stream:
        json.dump(obj, stream, separators=(',', ':'))
    pending.replace(path)


class ContactAudit:
    """Evaluation-only early physical contact, used after policy selection."""

    def begin_audit(self):
        self.audit = True
        self.audit_contacts = []
        self.audit_crossings = []
        self.active_pairs = set()
        self.previous_distance = None

    def _judge_substep(self, incoming):
        if getattr(self, 'audit', False) and self._steps < LOOK:
            data = self._data
            blade = data.geom_xmat[self._blade].reshape(3, 3)[:, 2]
            if blade[2] < 0:
                blade = -blade
            relative = self._ball_position - data.geom_xpos[self._blade]
            above = float(relative @ blade)
            edge = float(np.linalg.norm(relative - above * blade) - self._model.geom_size[self._blade, 0])
            distance = above - BALL_RADIUS - float(self._model.geom_size[self._blade, 1])
            if self.previous_distance is not None and self.previous_distance > 0 >= distance and incoming < -.15:
                self.audit_crossings.append(dict(time=float(data.time), edge_mm=1000 * edge,
                                                 slope_z=float(blade[2]), incoming=float(incoming)))
            self.previous_distance = distance
            now = set()
            for c in data.contact:
                if c.efc_address < 0 or self._ball_geom not in c.geom:
                    continue
                other = int(next(geom for geom in c.geom if geom != self._ball_geom))
                now.add(other)
                if other not in self.active_pairs:
                    self.audit_contacts.append(dict(time=float(data.time), geom=self._model.geom(other).name,
                                                    incoming=float(incoming), edge_mm=1000 * edge,
                                                    alignment=abs(float(c.frame[:3] @ blade)),
                                                    face_normal_z=float(blade[2]), distance_mm=1000*distance))
            self.active_pairs = now
        return super()._judge_substep(incoming)


class AuditStock(ContactAudit, BounceArena):
    pass


class AuditWide(ContactAudit, WidePosture):
    pass


def project(arena, rgb):
    data, model = arena._data, arena._model
    camera = model.camera(WRIST_CAMERA).id
    delta = arena._ball_position - data.cam_xpos[camera]
    x, y, z = data.cam_xmat[camera].reshape(3, 3).T @ delta
    height, width = rgb.shape[:2]
    focal = .5 * height / np.tan(.5 * np.deg2rad(model.cam_fovy[camera]))
    if z >= -.0001:
        return None
    return np.array([x * focal / -z, y * focal / -z, focal*BALL_RADIUS / -z])


def image_trace(arena, observer, clean, obs):
    rgb = obs['rgb']
    ellipse = _ellipse(rgb)
    projection = project(arena, rgb)
    gaps = observer.gaps
    state = observer.observe(obs)
    truth = clean.observe(obs)
    return dict(pixels=ellipse.tolist() if ellipse is not None else None,
                projected=projection.tolist() if projection is not None else None,
                accepted=observer.gaps == gaps, observed=state.tolist(), truth=truth.tolist(),
                q=obs['joints'].tolist())


def archived_group(cls, rows, schedule, cap):
    arena = cls(view='wrist', width=192, height=144, max_steps=cap)
    result = []
    nominal = arena._model.opt.gravity.copy()
    try:
        for row in rows:
            seed, tape = row['seed'], row['actions']
            arena._model.opt.gravity[:] = nominal
            obs = arena.reset(seed=seed, split='held', render=True)
            clean, observer = CleanState(arena), load_public(FUSED_OUTPUT)
            arena.begin_audit()
            trace = []
            for tick, action in enumerate(tape):
                if schedule:
                    arena._model.opt.gravity[:] = nominal * GRAVITIES[min(tick//SEGMENT, len(GRAVITIES)-1)]
                if tick < LOOK:
                    trace.append(image_trace(arena, observer, clean, obs))
                obs, _, done = arena.step(action, render=tick + 1 < LOOK)
                if done != (tick + 1 == len(tape)):
                    raise ValueError(f'archived tape terminal tick changed at {seed}')
            metrics = arena.judge_metrics()
            if metrics['longest_consecutive_rebounds'] != row.get('run', row.get('streak')):
                raise ValueError(f'archived rebound run changed at {seed}')
            if metrics['rebounds'] != row.get('score', row.get('rebounds')):
                raise ValueError(f'archived rebound count changed at {seed}')
            if schedule and [flight['contact_time'] for flight in metrics['rebound_events']] != row['impact_times']:
                raise ValueError(f'archived contact times changed at {seed}')
            result.append(dict(seed=seed, metrics=metrics, trace=trace, contacts=arena.audit_contacts,
                               crossings=arena.audit_crossings))
            print('archived', cls.__name__, seed, metrics['steps'], metrics['longest_consecutive_rebounds'],
                  metrics['other_collider'], flush=True)
    finally:
        arena.close()
    return result


def audit():
    switching = load(ROOT / 'continuous-gravity-optical.json.gz')
    posture = load(ROOT / 'posture-frozen.json.gz')
    output = dict(gravity=archived_group(AuditStock, switching['rows'], True, switching['cap']))
    save(ARCHIVED, output)
    for label, cls in (('ordinary', AuditStock), ('expanded', AuditWide)):
        output[label] = archived_group(cls, posture[label], False, 300)
        save(ARCHIVED, output)


class SupportedStartup:
    """Acquire a fully supported silhouette before trusting a clipped startup."""

    def __init__(self, sensor):
        self.sensor = sensor
        self.waiting = False
        self.confirmed = False
        self.supported = 0
        self.held_ticks = 0

    def observe(self, obs, tick):
        if self.confirmed or tick >= CLEAN_WINDOW:
            return self.sensor.observe(obs), False
        rgb, joints = obs['rgb'], obs['joints']
        height, width = rgb.shape[:2]
        ball = _ellipse(rgb)
        supported = ball is not None and np.all(abs(ball[:2]) + ball[2] + 1. < .5*np.array([width, height]))
        if supported:
            ball *= getattr(self.sensor.decoder, 'image_height', height) / height
        if not self.waiting and supported:
            self.confirmed = True
            return self.sensor.read(ball, joints), False
        self.waiting = True
        if self.sensor.home is None:
            self.sensor.home = self.sensor.decoder.body(joints)[12:15].copy()
        if supported:
            gaps = self.sensor.gaps
            candidate = self.sensor.read(ball, joints)
            self.supported = self.supported + 1 if self.sensor.gaps == gaps else 0
            if self.supported >= 2:
                self.confirmed = True
                return candidate, False
        else:
            self.sensor.previous = None
            self.sensor.history.clear()
            self.supported = 0
        self.held_ticks += 1
        return None, True


def run_live(arena, native, weights, seed, mode, schedule):
    nominal = arena._model.opt.gravity.copy()
    rendered = mode != 'full_state'
    obs = arena.reset(seed=seed, split='held', render=rendered)
    arena.begin_audit()
    clean = CleanState(arena) if mode in ('full_state', 'clean_prefix') else None
    observer = load_public(FUSED_OUTPUT) if rendered else None
    warm = SupportedStartup(observer) if mode == 'supported_startup' else None
    estimates = deque(maxlen=5)
    positions, visible = deque(maxlen=WINDOW), deque(maxlen=WINDOW)
    estimate = 1.
    issued = np.zeros(5)
    tape = []
    acquired = []
    for tick in range(arena.max_steps):
        ratio = GRAVITIES[min(tick//SEGMENT, len(GRAVITIES)-1)] if schedule else 1.
        arena._model.opt.gravity[:] = nominal * ratio
        hold = False
        if observer is not None:
            gaps = observer.gaps
            optical, hold = warm.observe(obs, tick) if warm is not None else (observer.observe(obs), False)
            if tick < LOOK:
                acquired.append(not hold and gaps == observer.gaps)
            if optical is None:
                positions.clear()
                visible.clear()
            elif schedule:
                positions.append(optical[:3].copy())
                visible.append(gaps == observer.gaps)
                if tick % 2 == 0:
                    measured = flight(positions, visible)
                    if measured is not None:
                        estimates.append(np.clip(-measured[0] / 9.28, .62, 1.40))
                        estimate = .65 * estimate + .35 * np.median(estimates)
        else:
            estimate = ratio
            optical = None
        if hold:
            action = np.zeros(5)
        else:
            state = clean.observe(obs) if mode == 'full_state' or mode == 'clean_prefix' and tick < CLEAN_WINDOW else optical
            state[3:6] /= np.sqrt(estimate)
            action = draw(native, state, weights, issued)
        tape.append(action.tolist())
        obs, _, done = arena.step(action, render=rendered)
        if done:
            break
    metrics = arena.judge_metrics()
    contacts, crossings = arena.audit_contacts, arena.audit_crossings
    arena.audit = False
    arena._model.opt.gravity[:] = nominal
    arena.reset(seed=seed, split='held', render=False)
    for tick, action in enumerate(tape):
        ratio = GRAVITIES[min(tick//SEGMENT, len(GRAVITIES)-1)] if schedule else 1.
        arena._model.opt.gravity[:] = nominal * ratio
        _, _, done = arena.step(action, render=False)
        if done != (tick + 1 == len(tape)):
            raise ValueError(f'{mode} replay ended on a different tick for seed {seed}')
    if digest_json(metrics) != digest_json(arena.judge_metrics()):
        raise ValueError(f'{mode} physical contact mismatch on seed {seed}')
    arena._model.opt.gravity[:] = nominal
    return dict(metrics=metrics, actions=tape, contacts=contacts, crossings=crossings,
                accepted=acquired, startup_hold_ticks=warm.held_ticks if warm else 0)


def held():
    with np.load(CHECKPOINT, allow_pickle=False) as checkpoint:
        weights = np.ascontiguousarray(checkpoint['weights'])
    archive = load(ARCHIVED)
    diagnosed = {row['seed'] for rows in archive.values() for row in rows}
    standard = fixed_seeds(12, 781602338)
    expanded = fixed_seeds(12, 781602339)
    if diagnosed.intersection(standard + expanded):
        raise ValueError('fresh held starts overlap the archived audit')
    native = Native()
    output = dict(clean_prefix_ticks=CLEAN_WINDOW, groups={})
    try:
        for name, cls, seeds, schedule, cap in (
                ('fresh_standard', AuditStock, standard, True, 480),
                ('fresh_expanded', AuditWide, expanded, False, 300),
                ('archived_gravity_failures', AuditStock,
                 [r['seed'] for r in archive['gravity'] if r['metrics']['steps'] < SEGMENT], True, 480)):
            records = output['groups'][name] = []
            arena = cls(view='wrist', width=192, height=144, max_steps=cap)
            try:
                for seed in seeds:
                    modes = {}
                    for mode in ('wrist', 'clean_prefix', 'supported_startup', 'full_state'):
                        modes[mode] = run_live(arena, native, weights, seed, mode, schedule)
                    records.append(dict(seed=seed, modes=modes))
                    save(HELD, output)
                    print(name, seed, [(mode, row['metrics']['longest_consecutive_rebounds'],
                                        row['metrics']['steps'], row['startup_hold_ticks'])
                                       for mode, row in modes.items()], flush=True)
            finally:
                arena.close()
    finally:
        native.working.cleanup()


def reachability():
    """Check launch feasibility with the existing privileged upper-bound controller."""
    audit_rows = load(ARCHIVED)
    groups = dict(gravity=[r for r in audit_rows['gravity'] if r['metrics']['steps'] < SEGMENT],
                  expanded=audit_rows['expanded'])
    report = dict(privileged=True, settings=dict(height_m=.065, regularization=.26, rate_limit=.65), groups={})
    settings = CeilingSettings(height=.065, regularization=.26, rate_limit=.65)
    for name, cls in (('gravity', AuditStock), ('expanded', AuditWide)):
        arena = cls(view='none', max_steps=SEGMENT)
        try:
            records = report['groups'][name] = []
            for row in groups[name]:
                seed = row['seed']
                arena.reset(seed=seed, split='held', render=False)
                controller = PrivilegedCeiling(arena, settings)
                tape = []
                for _ in range(arena.max_steps):
                    action = controller.command()
                    tape.append(action.tolist())
                    _, _, done = arena.step(action, render=False)
                    if done:
                        break
                metrics = arena.judge_metrics()
                arena.reset(seed=seed, split='held', render=False)
                for tick, action in enumerate(tape):
                    _, _, done = arena.step(action, render=False)
                    if done != (tick + 1 == len(tape)):
                        raise ValueError('reachability replay ended on a different tick')
                if digest_json(metrics) != digest_json(arena.judge_metrics()):
                    raise ValueError('reachability contact replay changed')
                records.append(dict(seed=seed, metrics=metrics, actions=tape))
            print('privileged launch feasibility', name,
                  [r['metrics']['longest_consecutive_rebounds'] for r in records], flush=True)
        finally:
            arena.close()
    save(REACHABILITY, report)


def figure():
    """Show actual detector behavior for an interior and a clipped startup."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    source = load(ROOT / 'continuous-gravity-optical.json.gz')['rows']
    diagnosis = load(ARCHIVED)['gravity']
    settings = ((2123523884, 'Survived 30 s'), (997103512, 'Lost after one rebound'))
    ticks = (1, 6, 9)
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 4.7), layout='constrained')
    fig.suptitle('Wrist startup on two archived stock drops')
    arena = BounceArena(view='wrist', width=192, height=144)
    try:
        for ix, (seed, outcome) in enumerate(settings):
            tape = next(row['actions'] for row in source if row['seed'] == seed)
            trace = next(row['trace'] for row in diagnosis if row['seed'] == seed)
            obs = arena.reset(seed=seed, split='held')
            for tick in range(ticks[-1] + 1):
                if tick in ticks:
                    ax = axes[ix, ticks.index(tick)]
                    ax.imshow(obs['rgb'], extent=(-96, 96, -72, 72))
                    for key, color, style in (('projected', '#68defa', '--'), ('pixels', '#ffe272', '-')):
                        circle = trace[tick][key]
                        if circle is not None:
                            ax.add_patch(Circle(circle[:2], circle[2], fill=False, lw=1.5,
                                                ec=color, linestyle=style))
                    ax.set(xlim=(-96, 96), ylim=(-72, 72), title=f'Tick {tick}')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if tick == ticks[0]:
                        ax.set_ylabel(outcome)
                obs, _, _ = arena.step(tape[tick])
            ax = axes[ix, -1]
            samples = list(range(min(12, len(trace))))
            errors = [1000 * np.linalg.norm(np.array(trace[t]['observed'][:3]) - trace[t]['truth'][:3])
                      for t in samples]
            ax.plot(samples, errors, marker='o', ms=3, color='#a75321')
            ax.set(xlim=(0, 11), ylim=(0, 360), xlabel='Tick', ylabel='Position error (mm)')
            ax.grid(alpha=.18)
        fig.supxlabel('Yellow solid shows the RGB fit. Cyan dashed shows evaluator-only projection.')
        fig.savefig(FIGURE, dpi=150)
    finally:
        plt.close(fig)
        arena.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--held', action='store_true', help='run causal interventions after the archived audit')
    modes.add_argument('--reachability', action='store_true', help='run the privileged physical upper-bound check')
    modes.add_argument('--figure', action='store_true', help='plot diagnostic frames from archived action tapes')
    cfg = parser.parse_args()
    if cfg.held:
        held()
    elif cfg.reachability:
        reachability()
    elif cfg.figure:
        figure()
    else:
        audit()


if __name__ == '__main__':
    main()
