"""Test causal transmission recall during the archived compound wrist course."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import WristPupil, _INC
from .tt24_compound_carry import BODIES, altered_wiring, templates
from .tt24_forward_wiring import (CALIBRATION, DIR, HISTORY, IDENTITY, clipped_error, inverse,
                                   load_forward, probe_batch, rating, ridge_fit, whiten)
from .tt24_motor_recall import sensor
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_public_morph import load_morph
from .tt24_view_recall import descriptor
from .tt24_world_arm import SOURCE


class Recall:
    """Choose a routing from ongoing pre-action encoder predictions."""

    def __init__(self, model, seed, *, probes=True):
        self.models = model
        self.rng = np.random.default_rng(seed)
        self.bank = [IDENTITY.copy()]
        self.choice = 0
        self.probes = probes
        self.discoveries = deque(maxlen=56)
        self.learning = False
        self.next_map = None
        self.pause = 0

    def reset(self, obs, body):
        self.model = self.models[body]
        self.states = deque((sensor(obs),), maxlen=HISTORY)
        self.commands = deque(maxlen=HISTORY-1)
        self.recent = deque(maxlen=6)
        self.records = []
        self.effective = []
        self.events = []
        self.probe_timing = []
        self.probe_queue = deque()
        self.pause = 0
        if self.learning and self.probes:
            self.queue_probes()

    def queue_probes(self):
        self.probe_queue = deque(probe_batch(self.rng, 20, scale=.34)[::2])

    def command(self, desired):
        chosen = self.bank[self.choice]
        action = desired if self.choice == 0 else inverse(chosen, desired)
        if self.probe_queue:
            self.probe_timing.append(len(self.records))
            action = .32*action + self.probe_queue.popleft()
        return np.clip(action, -.82, .82)

    def repair_history(self, pupil, new_map, age):
        old = np.array(self.effective[-age:])
        if not len(old):
            return
        corrected = np.clip(np.array(self.records[-age:]) @ new_map.T, -1., 1.)
        self.effective[-age:] = list(corrected)
        pupil.issued += _INC * np.sum(corrected - old, axis=0)

    def switch(self, i, pupil, age, tag):
        self.choice = i
        self.repair_history(pupil, self.bank[i], age)
        self.events.append(dict(tick=len(self.records), selection=int(i), event=tag))
        self.pause = 7

    def observe(self, request, obs, desired, pupil):
        after = sensor(obs)
        states = np.array(self.states)
        issued = np.vstack((*self.commands, request))
        row = whiten(*self.model.predict(states, issued), states[-1], after, self.model.noise)
        output = np.clip(self.bank[self.choice] @ request, -1., 1.)
        self.records.append(request.copy())
        self.effective.append(output)
        pupil.issued += _INC * (output - desired)
        self.commands.append(request.copy())
        self.states.append(after)
        self.recent.append(row)
        self.pause = max(0, self.pause-1)
        if len(self.recent) < 4:
            return
        losses = np.array([np.median([clipped_error(rate[i]) for rate in
                                    (rating(r, self.bank) for r in self.recent)])
                           for i in range(len(self.bank))])
        incumbent = losses[self.choice]
        likely = int(np.argmin(losses))
        better_known = likely != self.choice and losses[likely] < .74*incumbent and incumbent-losses[likely] > .30
        if better_known and self.pause == 0 and not self.learning:
            self.switch(likely, pupil, 5, 'recall')
            return

        if not self.learning:
            if self.pause or incumbent < 1.52 or np.linalg.norm(request) < .16:
                return
            self.learning = True
            self.discoveries.clear()
            self.discoveries.extend(list(self.recent)[-3:])
            self.next_map = None
            self.events.append(dict(tick=len(self.records), event='surprise', score=float(incumbent)))
            if self.probes:
                self.queue_probes()
            return

        self.discoveries.append(row)
        n = len(self.discoveries)
        # Guard against learning a new map for a transient impact or a bad image.
        if n >= 8 and incumbent < 1.08:
            self.learning = False
            self.probe_queue.clear()
            self.pause = 12
            self.events.append(dict(tick=len(self.records), event='cancel'))
            return
        if n < 10 or n % 2:
            return

        used = list(self.discoveries)
        current = self.bank[self.choice]
        raw = ridge_fit(used[:-3], prior=current, penalty=2.)
        u, s, v = np.linalg.svd(raw)
        stable = (u*np.clip(s, .40, 1.4)) @ v
        target_risks = rating(used[-1], self.bank + [stable])
        new_error = clipped_error(target_risks[-1])
        old_error = clipped_error(target_risks[self.choice])
        # Require unfit, strictly later observations to prefer the new map.
        later = np.array([rating(r, [current, stable]) for r in used[-3:]])
        signal = np.median([clipped_error(err[0])-clipped_error(err[1]) for err in later])
        if signal > .23 and new_error < old_error + .03:
            if self.next_map is not None:
                raw = ridge_fit(used, prior=stable, penalty=1.4)
                u, s, v = np.linalg.svd(raw)
                learned = (u*np.clip(s, .40, 1.4)) @ v
                self.bank.append(learned)
                self.switch(len(self.bank)-1, pupil, min(len(self.records), n), 'learn')
                self.learning = False
                self.next_map = None
                self.probe_queue.clear()
                return
            self.next_map = stable
        if n >= 40:
            self.learning = False
            self.probe_queue.clear()
            self.pause = 20
            self.events.append(dict(tick=len(self.records), event='abandon'))


class ConservativeRecall(Recall):
    """Select new routing only when it predicts multiple later motor responses."""

    def __init__(self, model, seed):
        super().__init__(model, seed, probes=False)
        self.rows = deque(maxlen=60)

    def reset(self, obs, body):
        super().reset(obs, body)
        self.rows.clear()
        self.vote = None

    def observe(self, request, obs, desired, pupil):
        after = sensor(obs)
        states = np.array(self.states)
        commands = np.vstack((*self.commands, request))
        row = whiten(*self.model.predict(states, commands), states[-1], after, self.model.noise)
        self.commands.append(request.copy())
        self.states.append(after)
        self.records.append(request.copy())
        interpreted = np.clip(self.bank[self.choice] @ request, -1., 1.)
        self.effective.append(interpreted)
        pupil.issued += _INC * (interpreted - desired)
        self.rows.append(row)
        self.pause = max(0, self.pause-1)
        if self.pause or len(self.rows) < 6 or len(self.records) % 2:
            return

        records = list(self.rows)
        test = records[-5:]
        past = records[:-5]
        candidates = self.bank.copy()
        sizes = (10, 22, 50)
        for width in sizes:
            if len(past) >= width:
                rough = ridge_fit(past[-width:], prior=self.bank[self.choice], penalty=2.8)
                left, singular, right = np.linalg.svd(rough)
                candidates.append((left*np.clip(singular, .45, 1.4)) @ right)
        sample = np.array([rating(x, candidates) for x in test])
        losses = np.sqrt(np.mean(np.minimum(sample, 6.)**2, axis=2))
        # Ignore single abrupt paddle impacts when assessing a persistent routing.
        robust = np.mean(np.sort(losses, axis=0)[1:-1], axis=0)
        chosen = int(np.argmin(robust))
        incumbent = robust[self.choice]
        known = chosen < len(self.bank)
        absolute, relative = (.18, .82) if known else (.30, .75)
        passes = (chosen != self.choice and incumbent-robust[chosen] > absolute
                  and robust[chosen] < relative*incumbent
                  and np.count_nonzero(losses[:, chosen] < losses[:, self.choice]) >= 4)
        consecutive = self.vote is not None and self.vote == (chosen if known else -1)
        self.vote = (chosen if known else -1) if passes else None
        if not passes or not consecutive:
            return
        if known:
            self.switch(chosen, pupil, 6, 'recall')
        else:
            fresh = candidates[chosen]
            self.bank.append(fresh)
            self.switch(len(self.bank)-1, pupil, min(14, len(self.records)), 'learn')
        self.rows.clear()
        self.vote = None


def play(arenas, schedule, mix, references, controller):
    reports = []
    for index, (body, seed) in enumerate(zip(BODIES, schedule, strict=True)):
        arena = arenas[body]
        regime = 'changed' if body else 'standard'
        obs = arena.reset(seed=seed, split='held', regime=regime)
        recognized = int(np.argmax(np.max(references @ descriptor(obs['rgb']), axis=1)))
        pupil = WristPupil.load(SOURCE)
        pupil.sensor = load_morph() if recognized else load_public(FUSED_OUTPUT)
        controller.reset(obs, recognized)
        tape, requests = [], []
        selected = [controller.choice]
        while True:
            desired = pupil.command(obs)
            request = controller.command(desired)
            physical = mix @ request if altered_wiring(index, len(tape)) else request
            requests.append(request.tolist())
            tape.append(physical.tolist())
            obs, _, done = arena.step(physical)
            controller.observe(request, obs, desired, pupil)
            selected.append(controller.choice)
            if done:
                break
        judge = arena.judge_metrics()
        arena.reset(seed=seed, split='held', regime=regime, render=False)
        for physical in tape:
            arena.step(physical, render=False)
        if arena.judge_metrics() != judge:
            raise ValueError('forward-routing tape replay differed')
        reports.append(dict(seed=seed, body=body, observation_model=recognized,
                            run=judge['longest_consecutive_rebounds'], total=judge['rebounds'],
                            ticks=len(tape), requests=requests, actions=tape,
                            adaptation=controller.events.copy(), probe_frames=controller.probe_timing.copy(),
                            choices=selected, terminal=judge))
        print(index, 'result', judge['longest_consecutive_rebounds'], 'probes',
              len(controller.probe_timing), 'choice', controller.choice, flush=True)
    return reports


def totals(rows):
    regions = {'all': slice(None), 'initial': slice(None, 3), 'first_shift': slice(3, 4),
               'changed_later': slice(4, 10), 'restore_shift': slice(10, 11), 'after': slice(11, None)}
    return {part: dict(streak=sum(r['run'] for r in rows[sl]), rebounds=sum(r['total'] for r in rows[sl]))
            for part, sl in regions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=int, required=True)
    parser.add_argument('--model', type=Path, default=CALIBRATION)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-probes', action='store_true')
    parser.add_argument('--conservative', action='store_true')
    args = parser.parse_args()
    suffix = '-conservative' if args.conservative else '-no-probes' if args.no_probes else ''
    out = args.output or DIR/f'forward-wiring-compound-{args.stream}{suffix}.json.gz'
    archive = DIR/f'compound-observer-{args.stream}.json.gz'
    old = _read(archive)
    if tuple(old['bodies']) != BODIES:
        raise ValueError('qualification schedule changed')
    wiring = np.array(old['wiring'])
    models = load_forward(args.model)
    controller = (ConservativeRecall(models, old['seed']+5671209) if args.conservative
                  else Recall(models, old['seed']+5671209, probes=not args.no_probes))
    arenas = [BounceArena(view='wrist', width=192, height=144, max_steps=360, arm=arm)
              for arm in ('stock', 'alternate')]
    try:
        measurements = play(arenas, old['starts'], wiring, templates(arenas, archived=True), controller)
    finally:
        for arena in arenas:
            arena.close()
    results = {key: totals(rows) for key, rows in old['results'].items()} | {'forward': totals(measurements)}
    _write(out, dict(schema='tt24-forward-routing-compound-1', legacy=str(archive), legacy_sha256=_hash(archive),
                     model_sha256=_hash(args.model), actor_sha256=_hash(SOURCE), fresh_task_reward_updates=0,
                     schedule=old['starts'], stream=args.stream, wiring=wiring.tolist(), results=results,
                     final_maps=[m.tolist() for m in controller.bank], adaptation_probe_frames=
                     sum(len(r['probe_frames']) for r in measurements), measurements=measurements))
    print('compound comparison', args.stream, results, flush=True)


if __name__ == '__main__':
    main()
