"""Train an opt-in native Pong curriculum with an independent release height."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import blake2b
from multiprocessing import get_context
from time import perf_counter

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong

from .tt3_pong import _native_rallies
from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_pong import CAP, OPPONENT_SPEED, ROLE_MODEL, cases
from .tt22_duration_transfer import MACRO
from .tt22_motor_transfer import MotorPrimitive
from .tt23_abstain import SETTINGS, arrays, choose, kernel
from .tt23_anonymous_action import FixedMotor
from .tt23_anonymous_replication import TRAIN
from .tt23_anonymous_target import target_context
from .tt23_direction import CHOICES, NATIVE, NATIVE_OFFSET, paired_returns
from .tt23_optical_contact import OUT

PLAN = OUT / 'decoupled-pong-plan.json.gz'
RAW = OUT / 'decoupled-pong-practice.json.gz'
RESULT = OUT / 'decoupled-pong-native.json.gz'
ARM = OUT / 'decoupled-pong-arm.json.gz'
VALIDATION_SEED = 1260553041
OFFSETS = (1050000000, 1100000000, 1150000000)
ENVIRONMENTS = ('coupled', 'decoupled')
BANKS = {'original': OUT / 'anonymous-action-arm.json.gz', 'crossflight': OUT / 'crossflight-arm.json.gz',
         'abstain': OUT / 'abstain-arm.json.gz'}
MODULUS = 2147483647
MULTIPLIER = 48271
PERSON = b'tt23-y1'


class DecoupledPong(Pong):
    """A separate per-environment random stream changes only release height."""

    def __init__(self, seeds, config=Physics(), device='cpu'):
        split = [int.from_bytes(blake2b(str(int(seed)).encode(), digest_size=8, person=PERSON).digest(),
                                'little') % MODULUS for seed in seeds]
        self.height_rng = torch.tensor(split, dtype=torch.int64, device=device)
        super().__init__(seeds, config, device)

    @torch.no_grad()
    def reset(self, mask):
        super().reset(mask)
        self.height_rng[mask] = (self.height_rng[mask] * MULTIPLIER + 1).remainder(MODULUS)
        relative = self.height_rng[mask].double() / MODULUS
        self.ball[mask, 1] = (.2 + .6 * relative).float()


def frozen_plan():
    source = _read(TRAIN)
    held = _read(NATIVE)
    groups = [dict(offset=group['offset'], count=len(group['rows']['2']),
                   seeds=[row['case']['serve'] for row in group['rows']['2']])
              for group in source['sets'] if group['offset'] in OFFSETS]
    native = dict(offset=NATIVE_OFFSET, count=len(held['rows']), seeds=[row['seed'] for row in held['rows']])
    selected = groups + [native]
    starts = [seed for group in selected for seed in group['seeds']]
    if [g['offset'] for g in groups] != list(OFFSETS) or len(set(starts)) != len(starts):
        raise ValueError('source cohorts are incomplete or overlap')
    if any([c.serve for c in cases(g['count'],g['offset'])] != g['seeds'] for g in selected):
        raise ValueError('native schedule differs from its earlier receipt')
    return dict(schema='tt23-decoupled-pong-plan-1', sources={p.name:_hash(p) for p in (TRAIN, NATIVE, ROLE_MODEL)},
                cohorts=groups, held=native, options=list(CHOICES), native_speed=OPPONENT_SPEED, cap=CAP,
                release_height_salt=PERSON.decode(), settings=[list(setting) for setting in SETTINGS],
                confirmatory_arm_seed=VALIDATION_SEED)


def run_points(schedule, role, *, factory=None):
    indexed = {c.serve:[] for c in schedule}
    queue = [(c, option) for c in schedule for option in CHOICES]
    for begin in range(0, len(queue), 42):
        batch = queue[begin:begin+42]
        agents = [FixedMotor(role, c.exploration, option) for c, option in batch]
        options = {} if factory is None else {'game_factory': factory}
        played = _native_rallies([c for c, _ in batch], agents, max_ticks=CAP,
                                opponent_speed=OPPONENT_SPEED, **options)
        for (c, option), agent, output in zip(batch, agents, played, strict=True):
            optical = None if agent.context is None else agent.context.tolist()
            indexed[c.serve].append(dict(output, context=optical, option=option))
    rows = []
    for seed, outcomes in indexed.items():
        first = outcomes[0]
        if any(row['case'] != first['case'] or row['context'] != first['context'] or row['option'] != choice
               for row, choice in zip(outcomes, CHOICES, strict=True)):
            raise ValueError('native alternatives changed the scene through the choice cue')
        rows.append(dict(seed=seed, context=first['context'], options=outcomes))
    return rows


def collect(group):
    torch.set_num_threads(1)
    started = perf_counter()
    role = RoleModel(**_read(ROLE_MODEL))
    fresh = run_points(cases(group['count'], group['offset']), role, factory=DecoupledPong)
    print('decoupled native', group['offset'], len(fresh), round(perf_counter() - started, 1), 's', flush=True)
    return dict(group, rows=fresh)


def legacy_groups():
    source = _read(TRAIN)
    training = {}
    for cohort in source['sets']:
        if cohort['offset'] not in OFFSETS:
            continue
        trials = cohort['rows']
        restored = []
        for series in zip(*(trials[str(i)] for i in CHOICES), strict=True):
            seed, context = series[0]['case']['serve'], series[0]['context']
            if any(outcome['case']['serve'] != seed or outcome['context'] != context or outcome['option'] != i
                   for outcome, i in zip(series, CHOICES, strict=True)):
                raise ValueError('ordinary archives mismatch before movement')
            restored.append(dict(seed=seed, context=context, options=list(series)))
        training[cohort['offset']] = restored
    held = _read(NATIVE)['rows']
    held = [dict(row, options=[row['options'][i] for i in CHOICES]) for row in held]
    return training, held


def reset_audit(legacy):
    rng = np.random.default_rng(789958125)
    seeds = rng.integers(MODULUS, size=16384).tolist()
    ordinary, independent = Pong(seeds), DecoupledPong(seeds)
    velocity = ordinary.ball_velocity[:, 1].numpy().copy()
    height = [game.ball[:, 1].numpy().copy() for game in (ordinary, independent)]
    if not torch.equal(ordinary.ball_velocity, independent.ball_velocity):
        raise ValueError('the new curriculum changed the angle or serve direction')
    if not torch.equal(ordinary.rng, independent.rng):
        raise ValueError('separate y values advanced the original angle stream')
    masks = [torch.rand(len(seeds), generator=torch.Generator().manual_seed(i)) < .49 for i in range(4)]
    for mask in masks:
        stationary = independent.ball[~mask].clone()
        ordinary.reset(mask)
        independent.reset(mask)
        if (not torch.equal(ordinary.ball_velocity, independent.ball_velocity)
                or not torch.equal(ordinary.rng, independent.rng)
                or not torch.equal(stationary, independent.ball[~mask])):
            raise ValueError('masked resets changed the original angle stream or unmasked launch')
    role = RoleModel(**_read(ROLE_MODEL))
    original = cases(len(legacy[OFFSETS[0]]),OFFSETS[0])[::12]
    default = run_points(original, role)
    for observed, reference in zip(default, legacy[OFFSETS[0]][::12], strict=True):
        if observed != reference:
            raise ValueError('default native opening command, observation or terminal events changed')
    version = dict(verified_legacy_action_tapes=len(default) * len(CHOICES),
                   sample_seeds=len(seeds), matched_angle_stream=True)
    for curriculum, values in zip(ENVIRONMENTS, height, strict=True):
        version[curriculum] = dict(mean=float(values.mean()), std=float(values.std()),
                                  percentiles=np.quantile(values, (.05,.50,.95)).tolist(),
                                  velocity_height_correlation=float(np.corrcoef(velocity,values)[0,1]))
    return version


def array_group(offset, rows):
    x, ticks, seen = arrays(rows)
    return dict(offset=offset, x=x, ticks=ticks, seen=seen)


def behavior(rows, model=None):
    x, ticks, seen = arrays(rows)
    action = np.full(len(x), 3) if model is None else choose(model, x, seen)
    chosen = [row['options'][a - 2] for row, a in zip(rows, action, strict=True)]
    mixed = seen & (x[:, 0] * x[:, 1] < 0)
    hold_earns_more = (ticks[:, 1] > ticks[:, 0]) & (ticks[:, 1] > ticks[:, 2])

    def scores(group):
        returns = [bool(row['contacts'] and row['contacts'][0]['returned']) for row in chosen]
        return dict(points=int(group.sum()), seconds=float(ticks[np.arange(len(x)), action - 2][group].mean() / 120)
                    if group.any() else None, first_returns=int(np.sum(np.asarray(returns)[group])),
                    holds=int(np.sum(action[group] == 3)), center_strictly_longer=int(np.sum(hold_earns_more[group])),
                    held_when_center_strictly_longer=int(np.sum(hold_earns_more[group] & (action[group] == 3))))

    return dict(all=scores(np.ones(len(x), dtype=bool)), mixed=scores(mixed), action=action.tolist())


def cross_validate(training):
    reports = []
    for setting in SETTINGS:
        fold_gains = {}
        for env, source in training.items():
            folds = []
            for i, held in enumerate(source):
                fit = kernel([b for j, b in enumerate(source) if i != j], setting)
                choices = choose(fit, held['x'], held['seen'])-2
                selected = held['ticks'][np.arange(len(choices)), choices]
                gain = float(np.mean(selected - held['ticks'][:, 1]) / 120)
                folds.append(dict(offset=held['offset'], gain=gain))
            fold_gains[env] = folds
        average = float(np.mean([fold['gain'] for folds in fold_gains.values() for fold in folds]))
        reports.append(dict(setting=list(setting), fold_gains=fold_gains, average=average))
    best = max(reports, key=lambda r:(r['average'], r['setting'][1], -r['setting'][0]))
    return best['setting'], reports


def fit_native(*, workers):
    began = perf_counter()
    torch.set_num_threads(1)
    plan = frozen_plan()
    if PLAN.exists() and _read(PLAN) != plan:
        raise ValueError('native plan changed after the initial collection')
    _write(PLAN,plan)
    raw = _read(RAW) if RAW.exists() else dict(plan_sha256=_hash(PLAN), cohorts={})
    if raw['plan_sha256'] != _hash(PLAN):
        raise ValueError('native observations use a different plan')
    missing = [g for g in (*plan['cohorts'],plan['held']) if str(g['offset']) not in raw['cohorts']]
    if missing and workers == 1:
        for result in map(collect, missing):
            raw['cohorts'][str(result['offset'])] = result
            _write(RAW,raw)
    elif missing:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            for result in pool.map(collect, missing):
                raw['cohorts'][str(result['offset'])] = result
                _write(RAW,raw)
    legacy, held_legacy = legacy_groups()
    independent = {off:raw['cohorts'][str(off)]['rows'] for off in OFFSETS}
    held_decoupled = raw['cohorts'][str(plan['held']['offset'])]['rows']
    training = {env:[array_group(off, source[off]) for off in OFFSETS]
                for env,source in zip(ENVIRONMENTS,(legacy,independent),strict=True)}
    setting, validation = cross_validate(training)
    models = {name:kernel(groups,setting,shuffle=mode=='shuffle')
              for env,groups in training.items() for mode,name in (('true',env),('shuffle',env+'_shuffle'))}
    evaluations = {env:{name:behavior(rows, model) for name,model in dict(models,center=None).items()}
                   for env,rows in zip(ENVIRONMENTS,(held_legacy,held_decoupled),strict=True)}
    coverage = {env:[dict(offset=g['offset'], seen=int(g['seen'].sum()), mixed=int(np.sum(
                         g['seen'] & (g['x'][:, 0] * g['x'][:, 1] < 0)))) for g in groups]
                for env,groups in training.items()}
    held_cov = {env:int(np.sum((x[:,0] * x[:,1] < 0) & seen)) for env,(x,_,seen) in (
        (name,arrays(rows)) for name,rows in zip(ENVIRONMENTS,(held_legacy,held_decoupled),strict=True))}
    result = dict(schema='tt23-decoupled-pong-native-1', plan_sha256=_hash(PLAN), practice_sha256=_hash(RAW),
                  marginals=reset_audit(legacy), setting=setting, validation=validation, models=models,
                  native=evaluations, training_coverage=coverage, held_mixed=held_cov, wall_seconds=perf_counter()-began)
    _write(RESULT,result)
    print('native frozen',setting, 'training',coverage, 'held mixed',held_cov, flush=True)
    for env,scores in evaluations.items():
        print(env,{name:score['all'] for name,score in scores.items()},flush=True)


def score_arm(rows, models):
    ready = MotorPrimitive(_read(MACRO)['parameters'], None).ready_world
    context = [target_context(row['sight'],ready) for row in rows]
    seen = np.array([x is not None for x in context])
    x = np.array([x if x is not None else np.zeros(4) for x in context])
    is_mixed = seen & (x[:,0] * x[:,1] < 0)
    actions = {name:choose(model,x,seen) for name,model in models.items()}
    actions['center'] = np.full(len(rows),3)
    score = {name:np.array([bool(row['options'][idx]['legal']) for row,idx in zip(rows,choice,strict=True)])
             for name,choice in actions.items()}
    contacts = {name:np.array([bool(row['options'][idx]['contacts']) for row,idx in zip(rows,choice,strict=True)])
                for name,choice in actions.items()}
    groups = {}
    for group,mask in [('all',np.ones(len(rows),dtype=bool)),('optically_mixed',is_mixed),
                       ('opposite_release',np.array([r.get('family')=='opposite' for r in rows]))]:
        if not mask.any():
            continue
        scores = {name:dict(legal=int(np.sum(y[mask])), contacts=int(np.sum(contacts[name][mask])),
                           holds=int(np.sum(actions[name][mask]==3))) for name,y in score.items()}
        paired = {name:paired_returns(score['decoupled'][mask].tolist(),control[mask].tolist())
                  for name,control in score.items()
                  if name!='decoupled'}
        groups[group] = dict(points=int(mask.sum()),scores=scores,paired=paired)
    return dict(groups=groups, actions={name:choice.tolist() for name,choice in actions.items()},
                seeds=[r['seed'] for r in rows])


def assess_arm():
    source = _read(RESULT)
    if source['plan_sha256']!=_hash(PLAN) or source['practice_sha256']!=_hash(RAW):
        raise ValueError('the frozen native fit changed before arm evaluation')
    cohorts = {}
    for name,path in BANKS.items():
        bank = _read(path)
        rows = bank['rows']
        if len(rows) != 64 or (name != 'original' and bank.get('verified') != 192):
            raise ValueError('use a complete physical bank')
        cohorts[name] = score_arm(rows,source['models'])
    native = source['native']['decoupled']
    gate = native['decoupled']['all']['seconds'] > max(native['center']['all']['seconds'],
                                                       native['decoupled_shuffle']['all']['seconds'])
    mixed = {name:0 for name in ('decoupled','coupled','center')}
    for name in ('crossflight','abstain'):
        all_scores = cohorts[name]['groups']['all']['scores']
        active = all_scores['decoupled']['legal']
        gate &= active >= all_scores['coupled']['legal']+2 and active > all_scores['decoupled_shuffle']['legal']
        gate &= active >= all_scores['center']['legal']
        local = cohorts[name]['groups']['optically_mixed']['scores']
        for key in mixed:
            mixed[key] += local[key]['legal']
    gate &= mixed['decoupled'] >= mixed['coupled'] + 4 and mixed['decoupled'] >= mixed['center']
    report = dict(schema='tt23-decoupled-pong-arm-1', source_sha256=_hash(RESULT), banks={
        name:dict(fingerprint=_hash(BANKS[name]),**value) for name,value in cohorts.items()},
        pooled_optically_mixed=mixed, new_mixed_arm_gate=bool(gate))
    _write(ARM,report)
    for name,bank in cohorts.items():
        print(name,{g:row['scores'] for g,row in bank['groups'].items()},flush=True)
    print('new mixed arm gate',gate,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=1)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument('--native-only',action='store_true')
    phases.add_argument('--arm-only',action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('use a positive number of workers')
    if not args.arm_only:
        fit_native(workers=args.workers)
    if not args.native_only:
        assess_arm()


if __name__ == '__main__':
    main()
