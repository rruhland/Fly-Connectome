"""Accept a public encoder-driven inspection request before the same held-ball release."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from multiprocessing import get_context

import mujoco
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_continual_wrist import ContinualWrist
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm
from .tt24_motor_bank import I, MotorBank, decoder
from .tt24_ready import prepare_step
from .tt24_single_arm_course import CAP, STAGES, course_plan


INVISIBLE = 5
RESTORED = (7, 8, 11)
REPORT = ROOT / 'current-pause-summary.json.gz'


class Fixture:
    """Keep release and its safety interlock in hardware, outside the learner's port."""

    def __init__(self, arm, plan, launch, wire):
        self.arm = arm
        self.env = arm.arena
        self.plan = plan
        self.location, self.velocity, self.spin = launch
        self.wire = np.asarray(wire)
        self.changes = dict(plan['changes'])

    def start(self, render):
        self.holding = True
        self.live = 0
        self.release_index = None
        self.release_state = None
        self.events = []
        self.actions = []
        self.phases = []
        self.requests = []
        return self.arm.reset(self.plan, self.location, render)

    def release(self, render):
        if not self.holding:
            raise ValueError('release piston already actuated')
        self.holding = False
        self.release_index = len(self.actions)
        env = self.env
        env._ball_velocity[:] = self.velocity
        env._ball_spin[:] = self.spin
        mujoco.mj_forward(env._model, env._data)
        obs = env._observation(render)
        self.release_state = dict(joints=obs['joints'].tolist(), rates=obs['joint_velocity'].tolist(),
                                  time_s=float(env._data.time))
        if render:
            self.release_state['wrist_sha256'] = sha256(obs['rgb'].tobytes()).hexdigest()
        return obs

    def issue(self, request, category, render):
        command = np.clip(self.wire @ request, -1., 1.)
        return self.advance(command, category, render, request)

    def advance(self, physical, category, render, request=None):
        self.actions.append(np.asarray(physical).tolist())
        self.phases.append(category)
        self.requests.append(None if request is None else np.asarray(request).tolist())
        if self.holding:
            if category == 'playing':
                raise ValueError('playing command issued before release')
            prepare_step(self.env, np.asarray(physical), self.location)
            return self.env._observation(render), False
        if category != 'playing':
            raise ValueError('cannot issue contact-free inspection after release')
        if self.live in self.changes:
            self.events.append(self.arm.switch_gravity(self.live, self.changes[self.live]))
        obs, _, done = self.env.step(physical, render=render)
        self.live += 1
        return obs, done

    def offer_inspection(self, public_joints, learner, category):
        """Return nothing if the fixture cannot safely suspend this release."""
        if not self.holding:
            return None

        start = len(self.actions)

        def command_and_read(command):
            obs, _ = self.issue(command, category, False)
            return obs['joints']

        status = learner.inspect(public_joints, command_and_read, self.plan['calibration_rng'])
        end = len(self.actions)
        if end-start != status['total_ticks']:
            raise ValueError('the charged inspection length differs')
        return status | dict(start_tick=start, end_tick=end, request_kind=category)

    def score(self):
        return dict(actions=self.actions, phases=self.phases, requests=self.requests,
                    release_tick=self.release_index, release_state=self.release_state,
                    metrics=self.env.judge_metrics(), events=self.events, live_ticks=self.live)


def replay(fixture, record):
    fixture.start(False)
    for tick, (command, phase) in enumerate(zip(record['actions'], record['phases'], strict=True)):
        if tick == record['release_tick']:
            fixture.release(False)
        _, done = fixture.advance(command, phase, False)
        if done != (tick+1 == len(record['actions'])):
            raise ValueError('terminal step differs in ordered inspection replay')
    check = fixture.score()
    for key in ('actions','phases','release_tick','metrics','events','live_ticks'):
        if digest_json(check[key]) != digest_json(record[key]):
            raise ValueError(f'ordered mixed-phase replay changed {key}')
    if 'release_state' in record:
        bones = {k:v for k,v in record['release_state'].items() if k!='wrist_sha256'}
        if digest_json(bones) != digest_json(check['release_state']):
            raise ValueError('the actual release pose or clock differs')
    return True


def archived_row(plan, row):
    before = row['inspection']
    actions = row['actions']
    lead = plan['prep_ticks']
    return dict(actions=before+actions,
                phases=['opening_inspection']*len(before)+['setup']*lead+['playing']*(len(actions)-lead),
                release_tick=len(before)+lead, live_ticks=len(actions)-lead,
                metrics=row['metrics'], events=row['events'])


def save_memory(policy):
    return dict(maps=[m.tolist() for m in policy.bank.maps], selected=policy.bank.selected,
                pending=policy.monitor.pending, eye=policy.last_eye)


def restore_memory(data):
    policy = ContinualWrist()
    if data is None:
        return policy
    policy.bank = MotorBank(np.array(data['maps']))
    policy.bank.selected = data['selected']
    if data['selected'] is not None:
        policy.bank.inverse = decoder(policy.bank.maps[data['selected']])
    policy.monitor.pending = data['pending']
    policy.last_eye = data['eye']
    return policy


def execute(fixture, policy, *, inspect_current=True, equal=False):
    obs = fixture.start(True)
    opening = None
    if policy.open_release(obs):
        opening = fixture.offer_inspection(obs['joints'], policy, 'opening_inspection')
        obs = fixture.env._observation(True)
    policy.begin_play(obs)
    eye = policy.policy.calibration_index
    pause = None
    stale = None
    traces = []
    first_request_tick = None

    def observe(request, measured):
        nonlocal first_request_tick, obs, pause, stale
        error = policy.observe(request, measured)
        pending = policy.inspection_requested
        traces.append([len(fixture.actions)-1, error, bool(pending)])
        if inspect_current and pending and pause is None:
            if first_request_tick is None:
                first_request_tick = len(fixture.actions)
            old = policy.bank.selected
            report = fixture.offer_inspection(measured['joints'],policy,'surprise_inspection')
            if report is not None:
                stale = old
                pause = report | dict(before_map=old, after_map=policy.bank.selected)
                obs = fixture.env._observation(True)
                policy.begin_play(obs)

    for _ in range(fixture.plan['prep_ticks']):
        request = policy.act(obs)
        obs, _ = fixture.issue(request,'setup',True)
        observe(request, obs)
    obs = fixture.release(True)
    yoke = None
    if equal and pause is not None:
        yoke = deepcopy(policy)
        yoke.bank.selected = stale
        if stale is not None:
            yoke.bank.inverse = decoder(yoke.bank.maps[stale])
        yoke.monitor.current = I if stale is None else yoke.bank.maps[stale]
        yoke.monitor.start(obs['joints'])
    live_selected = policy.bank.selected
    while True:
        request = policy.act(obs)
        obs, done = fixture.issue(request,'playing',True)
        observe(request, obs)
        if done:
            break

    row = fixture.score() | dict(opening_inspection=opening, same_release_inspection=pause,
                                 calibration_index=int(eye), selection_at_release=live_selected,
                                 first_request_tick=first_request_tick, monitor_trace=traces,
                                 pending_at_end=policy.monitor.pending)
    estimated = I if live_selected is None else policy.bank.maps[live_selected]
    row['relative_wiring_error_at_release'] = float(np.linalg.norm(estimated-fixture.wire)/np.linalg.norm(fixture.wire))
    return row, yoke


def equal_start(fixture, setup, policy):
    """Evaluator control. Preserve all pre-release commands and the public controller history."""
    fixture.start(False)
    until = setup['release_tick']
    for action, phase in zip(setup['actions'][:until],setup['phases'][:until],strict=True):
        fixture.advance(action,phase,False)
    obs = fixture.release(True)
    if digest_json(fixture.release_state)!=digest_json(setup['release_state']):
        raise ValueError('equal-timing control did not start at the identical release')
    while True:
        request = policy.act(obs)
        obs,done=fixture.issue(request,'playing',True)
        if done:
            break
    return fixture.score() | dict(selection_at_release=policy.bank.selected,
                                 same_physical_setup=True, ignored_accepted_joint_identification=True)


def arm_group():
    result = {name:InstalledArm(name) for name in ('stock','alternate')}
    for arm in result.values():
        arm.arena.max_steps=CAP
    return result


def scene(arms, source, row, server):
    plan=row['plan']
    server.reset(seed=plan['launcher_seed'],split='held',render=False)
    launch=tuple(value.copy() for value in (server._ball_position,server._ball_velocity,server._ball_spin))
    return Fixture(arms[plan['body']],plan,launch,source['wires'][row['wire_index']])


def score(metrics):
    return {key:metrics[key] for key in ('rebounds','longest_consecutive_rebounds','elapsed_s','steps')}


def run(index):
    original = ROOT / f'single-arm-course-{index}.json.gz'
    output = ROOT / f'current-pause-stream-{index}.json.gz'
    original_data=_read(original)
    config, physical_wires = course_plan(index)
    if len(original_data['rows'])!=len(config) or len(config)!=len(STAGES) or not (
            np.array_equal(original_data['wires'],physical_wires)
            and all(digest_json(x['plan'])==digest_json(p) for x,p in zip(original_data['rows'],config,strict=True))):
        raise ValueError('the baseline schedules or wiring are unavailable')
    origin=dict(schema='tt24-current-pause-1', original=original.name, original_sha256=_hash(original),
                wrist_bundle_sha256=_hash(BUNDLE), stream=index,
                trigger='unchanged public MotorMemory, pending accepted by fixture during held setup only',
                reference='same originally scheduled count of normal visual setup ticks, plus charged probes',
                rows=[], controller=None)
    data=_read(output) if output.exists() else origin
    if any(data[k]!=v for k,v in origin.items() if k not in ('rows','controller')):
        raise ValueError('the resumed qualification changed inputs')
    if [r['plan']['learner_seed'] for r in data['rows']]!=[x['learner_seed'] for x in config[:len(data['rows'])]]:
        raise ValueError('resume release arguments differ')
    policy=restore_memory(data['controller'])
    arms=arm_group()
    server=BounceArena(view='none')
    try:
        for i,prior in enumerate(original_data['rows'][len(data['rows']):],len(data['rows'])):
            physical=scene(arms,original_data,prior,server)
            baseline={}
            for name,record in prior['modes'].items():
                tape=archived_row(prior['plan'],record)
                replay(physical,tape)
                baseline[name]=score(tape['metrics']) | dict(release_tick=tape['release_tick'],
                                       inspection_ticks=len(record['inspection']),verified=True)
            measured,yoke=execute(physical,policy,equal=i in (INVISIBLE,*RESTORED))
            replay(physical,measured)
            temporal_control=None
            if yoke is not None:
                temporal_control=equal_start(physical,measured,yoke)
                replay(physical,temporal_control)
            data['rows'].append(dict(plan=prior['plan'],wire_index=prior['wire_index'],
                                          fixed_release=baseline,current=measured,yoked_stale=temporal_control,
                                          exact_replay=True))
            data['controller']=save_memory(policy)
            _write(output,data)
            pause=measured['same_release_inspection']
            print(index,i,'rebounds',measured['metrics']['rebounds'],
                  [baseline[s]['rebounds'] for s in ('adaptive','frozen','known')],
                  'pause',None if pause is None else pause['total_ticks'],
                  'stale',None if temporal_control is None else temporal_control['metrics']['rebounds'],flush=True)
        if 'source_retention' not in data:
            replayed=scene(arms,original_data,original_data['rows'][1],server)
            pristine,_=execute(replayed,ContinualWrist(),inspect_current=False)
            older=archived_row(original_data['rows'][1]['plan'],original_data['rows'][1]['modes']['adaptive'])
            if digest_json(pristine['actions'])!=digest_json(older['actions']) or digest_json(
                    pristine['metrics'])!=digest_json(older['metrics']):
                raise ValueError('recalled frozen bundle changed the original stock trajectory')
            replay(replayed,pristine)
            checkpoint=ROOT/f'current-pause-stream-{index}.npz'
            policy.save_between_releases(checkpoint)
            if save_memory(ContinualWrist(checkpoint))!=data['controller']:
                raise ValueError('the retained learned map bank failed exact reload')
            data['source_retention']=dict(stock_recall=pristine, exact_archive=True,
                                      final_memory_checkpoint=checkpoint.name,
                                      final_memory_sha256=_hash(checkpoint), exact_memory_restore=True)
            _write(output,data)
    finally:
        server.close()
        for arm in arms.values():
            arm.arena.close()
    if _hash(BUNDLE)!=origin['wrist_bundle_sha256']:
        raise ValueError('original carried readout changed')
    return output


def summarize(records):
    summary={}
    for label, indices in [('full',tuple(range(len(STAGES)))),('invisible_switch',(INVISIBLE,)),
                             ('restored_body',RESTORED)]:
        rows=[run['rows'][i] for run in records for i in indices]
        this=dict(drops=len(rows),current_rebounds=sum(r['current']['metrics']['rebounds'] for r in rows),
                  current_streak_sum=sum(r['current']['metrics']['longest_consecutive_rebounds'] for r in rows),
                  late_inspections=sum(r['current']['same_release_inspection'] is not None for r in rows),
                  added_inspection_ticks=sum((r['current']['same_release_inspection'] or {}).get('total_ticks',0) for r in rows),
                  release_ticks=[r['current']['release_tick'] for r in rows],
                  fixed={mode:sum(r['fixed_release'][mode]['rebounds'] for r in rows)
                         for mode in ('adaptive','frozen','known')},
                  zero_calibration_drop_rewards=True,
                  yoked=[dict(seed=r['plan']['learner_seed'],calibrated=r['current']['metrics']['rebounds'],
                              stale=r['yoked_stale']['metrics']['rebounds']) for r in rows if r['yoked_stale'] is not None])
        this['added_inspection_seconds']=this['added_inspection_ticks']/60.
        this['current_all_held_seconds'] = sum(r['current']['release_tick'] for r in rows)/60.
        this['fixed_adaptive_all_held_seconds'] = sum(r['fixed_release']['adaptive']['release_tick']
                                                     for r in rows)/60.
        this['current_opening_inspection_seconds'] = sum((r['current']['opening_inspection'] or {}).get(
            'total_ticks',0) for r in rows)/60.
        summary[label]=this
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--stream',nargs='+',type=int,choices=(0,1,2),default=(0,1,2))
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        paths=list(pool.map(run,args.stream))
    records=[_read(p) for p in paths]
    report=summarize(records)
    rows = [row for stream in records for row in stream['rows']]
    yoked_controls = sum(row['yoked_stale'] is not None for row in rows)
    denied = 0
    for row in rows:
        current = row['current']
        request = current['first_request_tick']
        denied += current['same_release_inspection'] is None and request is not None and request>current['release_tick']
    audit = dict(yoked_controls=yoked_controls, baseline_replays=sum(len(row['fixed_release']) for row in rows),
                 verified_new=len(rows)+yoked_controls, verified_pristine_recalls=len(records),
                 late_denials_without_pause=denied,
                 pristine_recall_rebounds=[x['source_retention']['stock_recall']['metrics']['rebounds'] for x in records])
    _write(REPORT,dict(schema='tt24-current-pause-summary-1', summary=report,
                       streams={p.name:_hash(p) for p in paths},
                       original_bundle_sha256=_hash(BUNDLE), audit=audit))
    print(report,flush=True)


if __name__=='__main__':
    main()
