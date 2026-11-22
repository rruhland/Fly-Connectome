"""Adapt source direction and holding decisions from selected physical point durations."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import betaincinv

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import MACRO
from .tt22_motor_transfer import MotorPrimitive
from .tt23_anonymous_target import target_context
from .tt23_direction import CHOICES, NATIVE, SOURCE, pick
from .tt23_optical_contact import OUT

TRAIN = {'ordinary_first': OUT / 'anonymous-action-arm.json.gz', 'ordinary_second': OUT / 'anonymous-replication-arm.json.gz',
         'independent': OUT / 'crossflight-arm.json.gz'}
HELD = {'ordinary': OUT / 'smooth-jog-arm.json.gz', 'independent': OUT / 'abstain-arm.json.gz'}
RESULT = OUT / 'adaptive-direction.json.gz'
PRACTICE = OUT / 'adaptive-direction-plan.json.gz'
DRAWS = 256
SEED = 37979561
BUDGET = np.arange(0, 49, 6)
ORDERS = {'ordinary_first': ('o1', 'i1', 'i2', 'o2'), 'independent_first': ('i1', 'o1', 'o2', 'i2')}
MODES = ('warm', 'flat', 'frozen', 'center')
CENTER = 1
PSEUDO_COUNT = 2.
BIAS = .10
OPTIONS = ('left', 'center', 'right')


def action_in_frame(raw, reflected):
    return np.where(reflected, 2 - raw, raw)


def encode(contexts, native):
    seen = np.array([x is not None for x in contexts])
    x = np.asarray([np.zeros(4) if x is None else x for x in contexts])
    mixed = (x[:, 0] * x[:, 1] < 0).astype(int)
    reflection = x[:, 0] < 0
    nominal = np.array([pick(c,native,'learned') - 2 for c in contexts])
    preferred = action_in_frame(nominal, reflection)
    difference = np.eye(3)[preferred] - np.eye(3)[2 - preferred]
    difference[preferred == CENTER] = (-.5, 1., -.5)
    return dict(seen=seen, group=mixed, reflect=reflection, nominal=nominal, hint=BIAS * difference)


def terminal_rank(elapsed, past):
    less = sum(t < elapsed for t in past)
    equal = sum(t == elapsed for t in past)
    return (.5 + less + .5 * equal) / (1. + len(past))


def parameters(encoding, wins, mass, warmed):
    group = encoding['group']
    prior = (.5 + encoding['hint']) if warmed else .5
    return PSEUDO_COUNT * prior + wins[group], PSEUDO_COUNT + mass[group]


def predict(encoding, wins, mass, warmed, ties):
    alpha, total = parameters(encoding, wins, mass, warmed)
    canonical = np.argmax(alpha / total + ties, axis=1)
    raw = action_in_frame(canonical, encoding['reflect'])
    return np.where(encoding['seen'], raw, CENTER)


def learn(elapsed, encoding, order, uniforms, warmed):
    """Consume one chosen outcome after each decision. Other outcomes stay in the bank."""
    successes, count = np.zeros((2,3)), np.zeros((2,3))
    observed = [[], []]
    trace = []
    snapshots = [(successes.copy(), count.copy())]
    prior = .5 + encoding['hint'] if warmed else np.full((len(elapsed),3), .5)
    for step, row in enumerate(order, 1):
        group = encoding['group'][row]
        alpha = PSEUDO_COUNT * prior[row] + successes[group]
        total = PSEUDO_COUNT + count[group]
        drawn = int(np.argmax(betaincinv(alpha, total-alpha, uniforms[row])))
        seen = encoding['seen'][row]
        raw = 2-drawn if encoding['reflect'][row] else drawn
        if not seen:
            raw = CENTER
        length = int(elapsed[row, raw])
        reward = None
        if seen:
            reward = terminal_rank(length, observed[group])
            successes[group, drawn] += reward
            count[group, drawn] += 1.
            observed[group].append(length)
        trace.append(dict(index=int(row), choice=raw, frames=length, duration_rank=reward))
        if step in BUDGET:
            snapshots.append((successes.copy(),count.copy()))
    return snapshots, trace


def block(rows, cell):
    counts = {}
    found = []
    for row in rows:
        group = cell(row)
        if counts.get(group, 0) >= 3:
            continue
        found.append(row)
        counts[group] = counts.get(group,0) + 1
    if sorted(counts.values()) != [3,3,3,3]:
        raise ValueError('each block requires three launches in all four matched cells')
    return found


def load_bank(path, *, strict=False):
    bank = _read(path)
    rows = bank['rows']
    if len(rows) != 64 or len({row['seed'] for row in rows}) != 64:
        raise ValueError('expected a complete physical opening bank')
    if strict and bank.get('verified') != 192:
        raise ValueError('the mixed bank must be replay-verified')
    if any(any(row['options'][i] is None for i in CHOICES) for row in rows):
        raise ValueError('each start requires all three physical actions')
    return rows


def training_blocks():
    rows = {name:load_bank(path,strict=name=='independent') for name,path in TRAIN.items()}

    def ordinary(row):
        return row['domain'], row['lane'] > 0

    independent = rows['independent']
    i1 = block(independent, lambda row:tuple(row['signs']))
    used = {row['seed'] for row in i1}
    i2 = block([row for row in independent if row['seed'] not in used], lambda row:tuple(row['signs']))
    return dict(o1=block(rows['ordinary_first'],ordinary),o2=block(rows['ordinary_second'],ordinary),i1=i1,i2=i2)


def opening_inputs(rows, ready, native):
    contexts = [target_context(row['sight'],ready) for row in rows]
    payload = [[row['options'][i] for i in CHOICES] for row in rows]
    return dict(encoding=encode(contexts,native), seeds=[row['seed'] for row in rows],
                elapsed=np.array([[x['ticks'] for x in option] for option in payload]),
                legal=np.array([[x['legal'] > 0 for x in option] for option in payload]))


def native_inputs(native):
    bank = _read(NATIVE)
    if bank['source_sha256'] != _hash(SOURCE) or len(bank['rows']) != 48:
        raise ValueError('native retention requires the original disjoint holdout')
    vectors = []
    for row in bank['rows']:
        values = [row['options'][i] for i in CHOICES]
        vectors.append([(x['ticks'], bool(x['contacts'] and x['contacts'][0]['returned'])) for x in values])
    return dict(encoding=encode([row['context'] for row in bank['rows']],native),
                seeds=[row['seed'] for row in bank['rows']],
                elapsed=np.array([[r[0] for r in rr] for rr in vectors]),
                legal=np.array([[r[1] for r in rr] for rr in vectors]))


def full_point_inputs(native):
    paths = [OUT / f'jog-bank-{i}.json.gz' for i in range(1,5)]
    rows = []
    for path in paths:
        receipt = _read(path)
        if not receipt['complete'] or receipt['verified'] != 24 or len(receipt['cases']) != 8:
            raise ValueError('every full-rally bank must be complete and verified')
        rows.extend(receipt['cases'])
    payload = [[row['results'][name] for name in OPTIONS] for row in rows]
    return dict(seeds=[row['seed'] for row in rows], encoding=encode([r['context'] for r in rows],native),
                paths={path.name:_hash(path) for path in paths},
                outcomes={key:np.array([[read(x) for x in option] for option in payload])
                          for key,read in (('near',lambda r:r['outcome']['legal_returns']),
                                           ('rally',lambda r:r['longest']),
                                           ('open',lambda r:r['opening_return']))})


def make_plan(blocks, held):
    training = {tag:[row['seed'] for row in rows] for tag,rows in blocks.items()}
    flat = sum(training.values(),[])
    tests = sum((rows['seeds'] for rows in held.values()),[])
    if len(flat)!=len(set(flat)) or len(tests)!=len(set(tests)) or set(flat)&set(tests):
        raise ValueError('practice and held launches overlap')
    plan = dict(schema='tt23-adaptive-direction-plan-1', mode_seed=SEED, draws=DRAWS, budgets=BUDGET.tolist(),
                practice=training, held={name:bank['seeds'] for name,bank in held.items()},
                orders={name:list(order) for name,order in ORDERS.items()},
                prior_count=PSEUDO_COUNT, prior_bias=BIAS,
                inputs={p.name:_hash(p) for p in (*TRAIN.values(),*HELD.values(),SOURCE,NATIVE,MACRO)})
    if PRACTICE.exists() and _read(PRACTICE)!=plan:
        raise ValueError('adaptive comparison plan changed')
    _write(PRACTICE,plan)
    return plan


def record_for(bank, mode, learning, ties):
    n = len(bank['seeds'])
    if mode == 'center':
        return np.full((len(BUDGET), n),CENTER)
    if mode == 'frozen':
        return np.repeat(bank['encoding']['nominal'][None,:],len(BUDGET),axis=0)
    return np.array([predict(bank['encoding'],alpha,count,mode=='warm',ties)
                     for alpha,count in learning])


def summary(traces, choices, training, held):
    result = {}
    for mode in MODES:
        arm = {}
        for name,bank in held.items():
            picked = np.asarray(choices[mode][name])
            n = len(bank['seeds'])
            legal = bank['legal'][np.arange(n),picked].sum(axis=-1).mean(axis=0)
            duration = bank['elapsed'][np.arange(n),picked].mean(axis=(0,2))
            arm[name] = dict(legal=legal.tolist(), holds=(picked==CENTER).sum(axis=-1).mean(axis=0).tolist(),
                             seconds_per_point=(duration/(120 if name=='native' else 30)).tolist())
        events = traces[mode]
        elapsed = np.cumsum(np.array([[x['frames'] for x in event] for event in events]),axis=1)/30.
        practice_moves = np.array([[x['choice'] for x in event] for event in events])
        indexes = np.array([[x['index'] for x in event] for event in events])
        legal = training['legal'][indexes,practice_moves].cumsum(axis=1)
        result[mode] = dict(held=arm, seconds=np.column_stack((np.zeros(len(elapsed)),elapsed))[:,BUDGET].mean(axis=0).tolist(),
                            legal_practice=np.column_stack((np.zeros(len(elapsed)),legal))[:,BUDGET].mean(axis=0).tolist())
    return result


def simulate(training, held, order, draws):
    traces = {name:[] for name in MODES}
    choices = {name:{domain:[] for domain in held} for name in MODES}
    states = {name:[] for name in MODES}
    for draw in draws:
        rng = np.random.default_rng(draw)
        random = rng.uniform(.00001,.99999,size=(len(order),3))
        ties = 1.e-10*rng.random(3)
        for name in MODES:
            if name in ('warm','flat'):
                evolution, trace = learn(training['elapsed'],training['encoding'],order,random,name=='warm')
            else:
                evolution = []
                picks = np.full(len(order),CENTER) if name=='center' else training['encoding']['nominal'][order]
                trace = [dict(index=int(row), choice=int(j), frames=int(training['elapsed'][row,j]), duration_rank=None)
                         for row,j in zip(order,picks,strict=True)]
            for domain,bank in held.items():
                choices[name][domain].append(record_for(bank,name,evolution,ties))
            traces[name].append(trace)
            states[name].append((evolution[-1] if evolution else None,ties))
    return dict(summary=summary(traces,choices,training,held),trace=traces,
                choices={name:{d:np.asarray(out).tolist() for d,out in b.items()} for name,b in choices.items()}),states


def full_score(native, experiments, known):
    bank = full_point_inputs(native)
    if set(known)&set(bank['seeds']) or len(set(bank['seeds']))!=32:
        raise ValueError('full-rally evaluation shares practice or prior held launches')
    result = {}
    for order,states in experiments.items():
        result[order]={}
        for mode,trials in states.items():
            if mode in ('warm','flat'):
                picked = np.asarray([predict(bank['encoding'],*state,mode=='warm',ties) for state,ties in trials])
            else:
                fixed = np.full(32,CENTER) if mode=='center' else bank['encoding']['nominal']
                picked = np.repeat(fixed[None,:],len(trials),axis=0)
            value = {name:x[np.arange(32),picked] for name,x in bank['outcomes'].items()}
            result[order][mode]=dict(near=float(value['near'].sum(axis=1).mean()),
                                     opening=float(value['open'].sum(axis=1).mean()),
                                     ten=float((value['rally']>=10).sum(axis=1).mean()),
                                     twenty=float((value['rally']>=20).sum(axis=1).mean()),
                                     mean_longest=float(value['rally'].max(axis=1).mean()),
                                     choices=picked.tolist())
    return dict(schedules=result,bank_hashes=bank['paths'],held=bank['seeds'])


def run(output):
    started = perf_counter()
    native = _read(SOURCE)['pooled']
    ready = MotorPrimitive(_read(MACRO)['parameters'],None).ready_world
    blocks = training_blocks()
    all_rows = sum(blocks.values(), [])
    training = opening_inputs(all_rows,ready,native)
    held = {name:opening_inputs(load_bank(path,strict=name=='independent'),ready,native) for name,path in HELD.items()}
    held['native'] = native_inputs(native)
    plan = make_plan(blocks,held)
    canonical = {seed:i for i,seed in enumerate(training['seeds'])}
    draws = np.random.default_rng(SEED).integers(2**31,size=DRAWS).tolist()
    trials, states = {}, {}
    for name,sequence in ORDERS.items():
        order = [canonical[row['seed']] for tag in sequence for row in blocks[tag]]
        trials[name], states[name] = simulate(training,held,np.asarray(order),draws)
    def last(mode,domain):
        return float(np.mean([trace['summary'][mode]['held'][domain]['legal'][-1] for trace in trials.values()]))
    endpoint = {mode:{domain:last(mode,domain) for domain in HELD} for mode in MODES}
    warm, flat, prior, center = (endpoint[name] for name in MODES)
    gate = (sum(warm.values()) >= sum(flat.values())+2
            and all(warm[d] >= center[d]-2 for d in HELD)
            and warm['independent'] >= prior['independent']+2
            and warm['ordinary'] >= prior['ordinary']-2)
    full = full_score(native,states,training['seeds']+sum(plan['held'].values(),[])) if gate else None
    result = dict(schema='tt23-adaptive-direction-1', plan_sha256=_hash(PRACTICE), draws=draws,
                  experiments=trials, endpoint=endpoint, full_point_gate=bool(gate), full_points=full,
                  wall_seconds=perf_counter()-started)
    _write(output,result)
    print('Opening endpoints',endpoint,'full-point gate',gate,flush=True)
    for name,result in trials.items():
        print(name,{mode:{d:row['legal'] for d,row in x['held'].items()} for mode,x in result['summary'].items()},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RESULT)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
