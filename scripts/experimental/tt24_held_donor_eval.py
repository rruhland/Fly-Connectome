"""Check the independently frozen held-donor head on real wrist rollouts."""

import argparse
from time import perf_counter

import mujoco
import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy
from .tt24_donor_trial import DESIGNS, ROOT, Native, World, _seeds
from .tt24_held_donor import CHECKPOINT, CPP, HELD, TRAINING, SourceWorld, measure
from .tt24_integrated_stress import BODIES, InstalledArm, episode, schedule, summary
from .tt24_launch_context import draw
from .tt24_ready import prepare_step
from .tt4_transfer_trial import _hash, _read, _write


ORDINARY = ROOT / 'held-donor-ordinary.json.gz'
DONOR_HELD = ROOT / 'held-donor-native-held.json.gz'
MODES = ('frozen', 'trained', 'appearance_router')
TARGET_CAP = 480
PER_BODY = 8
ROSTER = 1797279260


class ObservedBundle(CarriedWristPolicy):
    def __init__(self, matrices):
        super().__init__(BUNDLE)
        self.matrices = matrices
        self.mode = 'frozen'
        self.detections = []

    def begin_release(self, obs):
        super().begin_release(obs)
        key = self.mode if self.mode != 'appearance_router' else ('trained' if self.calibration_index else 'frozen')
        self.weights = self.matrices[key]
        self.selected = key
        self.detections.clear()

    def act(self, obs):
        previous = self.sensor.gaps
        command = super().act(obs)
        self.detections.append(self.sensor.gaps == previous)
        return command


class Installation(InstalledArm):
    def __init__(self, body, ordinary):
        super().__init__(body)
        self.arena.max_steps = 300 if ordinary else TARGET_CAP
        if ordinary:
            self.arena.sample_drop = BounceArena.sample_drop

    def reset(self, plan, location, render):
        super().reset(plan, location, render=False)
        if plan['prep_ticks'] == 0:
            self.arena._ball_velocity[:] = self.velocity
            self.arena._ball_spin[:] = self.spin
            mujoco.mj_forward(self.arena._model, self.arena._data)
        return self.arena._observation(render)


def model():
    report = _read(TRAINING)
    if not report['selected_before_target'] or _hash(CHECKPOINT) != report['checkpoint_sha256']:
        raise ValueError('physical testing needs the sealed donor checkpoint')
    if _hash(BUNDLE) != report['bundle_sha256']:
        raise ValueError('the shared carried calibration changed after source fitting began')
    with np.load(CHECKPOINT, allow_pickle=False) as archive:
        matrices = dict(frozen=archive['original'].copy(), trained=archive['trained'].copy())
    return report, matrices


def check_tape(arm, bundle, plan, launch, which):
    bundle.mode = which
    live = episode(arm, bundle, plan, launch, 'bundle')
    live['selection'] = bundle.selected
    lead = plan['prep_ticks']
    hits = np.flatnonzero(bundle.detections)
    pre_sight = hits[hits < lead]
    live['optical_acquisition'] = dict(first_supported=int(hits[0]) if len(hits) else None,
                                       last_supported_before_release=int(pre_sight[-1]) if len(pre_sight) else None,
                                       frames_supported_before_release=len(pre_sight),
                                       detected_near_release=bool(np.any(bundle.detections[max(0,lead-5):lead+5])))
    checked = episode(arm, bundle, plan, launch, 'bundle', replay=live['actions'])
    for key in ('metrics', 'changes_reached'):
        if digest_json(live[key]) != digest_json(checked[key]):
            raise ValueError(f'raw command replay changed {key} on {plan["learner_seed"]}')
    return live


def roster(ordinary):
    if ordinary:
        return [dict(body=body, learner_seed=seed, launcher_seed=seed, prep_ticks=0,
                     mass_ratio=1., drag_ratio=1., changes=[], fixture='above_initial_paddle', offset=[0.,0.,0.])
                for body, seeds in zip(BODIES, _seeds(ROSTER+1, 2, 6), strict=True) for seed in seeds]
    rows = schedule(seed=ROSTER, per_body=PER_BODY)
    counts = {body:0 for body in BODIES}
    rng = np.random.default_rng(ROSTER+19)
    for plan in rows:
        body = plan['body']
        plan['fixture'] = 'world_fixed' if counts[body] % 2 == 0 else 'body_served'
        counts[body] += 1
        plan['offset'] = rng.uniform([-.021,-.021,-.009],[.021,.021,.009]).tolist()
    return rows


def verify_roster(source, plans, ordinary):
    prior = _read(ROOT / 'integrated-stress.json.gz')
    seen = {p[key] for p in prior['schedule'] for key in ('learner_seed','launcher_seed')}
    source_groups = [*source['held_validation'], *source['ordinary_validation']]
    source_groups.extend(group for update in source['updates'] for group in update['physical_plans'])
    seen.update(row[key] for group in source_groups for row in group
                for key in ('seed','dispenser_seed') if row[key] is not None)
    other = roster(not ordinary)
    seen.update(row[key] for row in other for key in ('learner_seed','launcher_seed'))
    if seen.intersection(row[key] for row in plans for key in ('learner_seed','launcher_seed')):
        raise ValueError('intended held roster overlaps source or earlier target starts')


def acquisition(rows):
    comparisons = {}
    for mode in MODES:
        entries = [case['modes'][mode] for case in rows]
        streak = [entry['metrics']['longest_consecutive_rebounds'] for entry in entries]
        rebounds = [entry['metrics']['rebounds'] for entry in entries]
        comparisons[mode] = dict(first=sum(value>=1 for value in rebounds), second=sum(value>=2 for value in rebounds),
                                 ten=sum(value>=10 for value in streak),
                                 detected_near_release=sum(row['optical_acquisition']['detected_near_release'] for row in entries),
                                 changed_calibration=sum(row['model_index']==1 for row in entries),
                                 observed_first_s=[e['metrics']['rebound_events'][0]['contact_time']-
                                                   c['plan']['prep_ticks']/60.
                                                   for c,e in zip(rows,entries,strict=True) if e['metrics']['rebound_events']])
    frozen = np.array([row['modes']['frozen']['metrics']['longest_consecutive_rebounds'] for row in rows])
    for variant in ('trained','appearance_router'):
        other = np.array([row['modes'][variant]['metrics']['longest_consecutive_rebounds'] for row in rows])
        comparisons[variant]['versus_frozen'] = dict(wins=int((other>frozen).sum()), ties=int((other==frozen).sum()),
                                                     losses=int((other<frozen).sum()))
    return comparisons


def target(ordinary):
    source, matrices = model()
    plans = roster(ordinary)
    verify_roster(source, plans, ordinary)
    out = ORDINARY if ordinary else HELD
    if out.exists():
        result = _read(out)
        if result['checkpoint_sha256'] != source['checkpoint_sha256']:
            raise ValueError('existing physical evidence came from a different checkpoint')
        if result['plans'] != plans and digest_json(result['plans']) != digest_json(plans):
            raise ValueError('the intended held roster has changed')
    else:
        result = dict(checkpoint_sha256=source['checkpoint_sha256'], bundle_sha256=source['bundle_sha256'],
                      plans=plans, phase='immediate' if ordinary else 'held',
                      cap_ticks=300 if ordinary else TARGET_CAP, cases=[], physical_tapes=0)
    used = {r['plan']['learner_seed'] for r in result['cases']}
    bundle = ObservedBundle(matrices)
    installations = {body:Installation(body, ordinary) for body in BODIES}
    dispensers = {body:BounceArena(view='none', arm=body) for body in BODIES}
    try:
        for plan in plans:
            seed = plan['learner_seed']
            if seed in used:
                continue
            if plan['fixture']=='world_fixed':
                source_body = 'stock'
            else:
                source_body = plan['body']
            dispenser = dispensers[source_body]
            arm = installations[plan['body']]
            dispenser.reset(seed=plan['launcher_seed'], split='held', regime=arm.regime, render=False)
            launch = (dispenser._ball_position.copy() + np.asarray(plan['offset']),
                      dispenser._ball_velocity.copy(), dispenser._ball_spin.copy())
            arm.velocity, arm.spin = launch[1:]
            trials = {which:check_tape(arm, bundle, plan, launch, which) for which in ('frozen', 'trained')}
            if trials['frozen']['model_index'] != trials['trained']['model_index']:
                raise ValueError('the paired installations did not identify the same initial optical model')
            selected = 'trained' if trials['frozen']['model_index'] else 'frozen'
            if not any(c['plan']['body']==plan['body'] for c in result['cases']):
                routed = check_tape(arm,bundle,plan,launch,'appearance_router')
                if digest_json(routed) != digest_json(trials[selected]):
                    raise ValueError('the independently replayed public router differs from its selected matrix')
                result['physical_tapes'] += 1
            trials['appearance_router'] = trials[selected]
            result['cases'].append(dict(plan=plan, release_location=launch[0].tolist(), modes=trials))
            result['physical_tapes'] += 2
            _write(out, result)
            print('ordinary' if ordinary else 'held', plan['body'], plan['fixture'], seed,
                  [(which, row['metrics']['longest_consecutive_rebounds'], round(row['observed_seconds'],2))
                   for which,row in trials.items()], flush=True)
        result['summary'] = {}
        groups = {body:[row for row in result['cases'] if row['plan']['body']==body] for body in BODIES}
        for body, entries in groups.items():
            value = result['summary'][body] = dict(scores=summary(entries, MODES), acquisition=acquisition(entries))
            if not ordinary:
                value['fixtures'] = {f:acquisition([row for row in entries if row['plan']['fixture']==f])
                                     for f in ('world_fixed','body_served')}
        _write(out, result)
    finally:
        for arm in installations.values():
            arm.arena.close()
        for dispenser in dispensers.values():
            dispenser.close()


def source_tape(world, seed, matrix, *, replay=None):
    prepared, notes = world.plan([seed])
    arena = world.arena
    arena.max_steps = 360
    plan = prepared[1][0]
    nominal = arena._model.opt.gravity.copy()
    arena._model.opt.gravity[:] = nominal*plan[17]
    arena._drag_force_scale = plan[0]
    location = arena._ball_position.copy()
    gravity = {int(plan[18]):plan[19], int(plan[20]):plan[21]}
    lead = notes[0]['lead']
    issued, velocity = plan[12:17].copy(), arena._ball_velocity.copy()
    tape = []
    try:
        for tick in range(lead + arena.max_steps):
            released = tick - lead
            if released in gravity:
                arena._model.opt.gravity[:] = nominal*gravity[released]
            if replay is None:
                mujoco.mj_kinematics(arena._model, arena._data)
                data = arena._data
                normal = data.geom_xmat[arena._blade].reshape(3,3)[:,2]
                velocity = (1. - plan[25])*arena._ball_velocity + plan[25]*velocity
                state = np.r_[arena._ball_position-plan[3:6]+plan[22:25], velocity,
                              data.geom_xpos[arena._blade]-plan[3:6],
                              normal[:2] * (1. if normal[2] >= 0 else -1.)]
                command = draw(world.native, state, matrix, issued)
            else:
                command = np.asarray(replay[tick])
            tape.append(command.tolist())
            if tick < lead:
                prepare_step(arena, command, location)
                if tick+1 == lead:
                    arena._ball_velocity[:] = plan[6:9]
                    arena._ball_spin[:] = plan[9:12]
                    mujoco.mj_forward(arena._model, arena._data)
            else:
                _, _, done = arena.step(command,render=False)
                if replay is not None and done != (tick+1==len(replay)):
                    raise ValueError('independent Python source replay changed its terminal frame')
                if done:
                    break
        return dict(actions=tape, metrics=arena.judge_metrics())
    finally:
        arena._model.opt.gravity[:] = nominal


def donor_holdout():
    source, matrices = model()
    if DONOR_HELD.exists():
        current = _read(DONOR_HELD)
        if current['checkpoint_sha256'] != source['checkpoint_sha256']:
            raise ValueError('existing source holdout came from a different sealed actor')
        return
    native = Native(CPP)
    dispenser = BounceArena(view='none')
    carried = CarriedWristPolicy()
    worlds = [SourceWorld(native, i, carried, dispenser) for i in range(len(DESIGNS))]
    options = np.asarray(list(matrices.values()))
    sets = _seeds(ROSTER+500, len(worlds), 16)
    try:
        fresh = [world.plan(seeds) for world,seeds in zip(worlds,sets,strict=True)]
        results = measure(worlds, [row[0] for row in fresh], options, 360)
        repeated = measure(worlds, [row[0] for row in fresh], options, 360)
        if not np.array_equal(results,repeated):
            raise ValueError('same-controller independent held donor results changed')
        audit = []
        for i, (world, seeds) in enumerate(zip(worlds, sets, strict=True)):
            for j, weights in enumerate(options):
                played = source_tape(world, seeds[0], weights)
                replayed = source_tape(world, seeds[0], weights, replay=played['actions'])
                if digest_json(played)!=digest_json(replayed):
                    raise ValueError('independent Python source event replay disagrees')
                py = [played['metrics'][k] for k in
                      ('rebounds','longest_consecutive_rebounds','steps','paddle_contacts')]
                native_score = results[i,j,0]
                audit.append(dict(donor=i, policy=list(matrices)[j], seed=seeds[0], same_counts=
                                  np.array_equal(py,native_score), native_counts=native_score.tolist(), **played))
    finally:
        for world in worlds:
            world.close()
        dispenser.close()
        native.working.cleanup()
    original_report = _read(ROOT/'tt24_donor-refined.json.gz')
    prior = Native()
    old_results = []
    try:
        for i, design in enumerate(DESIGNS):
            world = World(prior, design)
            try:
                cases = original_report['held_donor'][i][:8]
                seeds = [row['seed'] for row in cases]
                scores = world.play(options,world.prepare(seeds,'held'),300)
                expected = np.asarray([[row['conditions']['clean']['metrics'][k] for k in
                                        ('rebounds','longest_consecutive_rebounds','steps','paddle_contacts')]
                                       for row in cases])
                if not np.array_equal(scores[0],expected):
                    raise ValueError('the frozen actor no longer matches its original donor source receipt')
                old_results.append(dict(seeds=seeds,scores=scores.tolist()))
            finally:
                world.close()
    finally:
        prior.working.cleanup()
    _write(DONOR_HELD, dict(checkpoint_sha256=source['checkpoint_sha256'], actual_mujoco_referee=True,
                           plans=[r[1] for r in fresh], scores=results.tolist(), historical_donor=old_results,
                           python_raw_replay_audit=audit))
    rows = results[...,1].transpose(1,0,2).reshape(2,-1)
    old = np.concatenate([np.asarray(r['scores'])[...,1] for r in old_results],axis=1)
    print('fresh held donor',[(float(r.mean()),int((r>=2).sum()),int((r>=10).sum()),int(r.max())) for r in rows],
          'historical no-prep retest', [(float(r.mean()),int((r>=10).sum())) for r in old], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ordinary', action='store_true')
    group.add_argument('--donors', action='store_true')
    args = parser.parse_args()
    if args.donors:
        donor_holdout()
    else:
        target(args.ordinary)


if __name__ == '__main__':
    main()
