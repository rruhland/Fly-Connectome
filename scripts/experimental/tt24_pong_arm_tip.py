"""Test one end-effector joystick for genuine motor excitation across ordinary Pong poses."""

import argparse

import numpy as np
import torch

from fly_connectome.pong import Physics,Pong

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena,_HOME
from .tt24_carried_wrist import BUNDLE,CarriedWristPolicy
from .tt24_donor_trial import ROOT
from .tt24_motor_bank import I,STEP
from .tt24_pong_arm import (HEIGHT,WIDTH,NATIVE_CAP,PARKED,ScreenPlayer,bounce,wirings)
from .tt24_public_body import FUSED_OUTPUT,load_public
from .tt24_ready import prepare_step
from .tt4_transfer_trial import _hash,_read,_write


SOURCE = ROOT/'pong-arm-tip-source.json.gz'
HELD = ROOT/'pong-arm-tip-held.json.gz'
FORENSICS = ROOT/'pong-arm-tip-forensics.json.gz'
RAIL = np.asarray([.64,.65,.40])
RAIL /= np.linalg.norm(RAIL)
MARKER = np.array([.060,0.,0.])
MOTOR_LIMIT = .267
GAIN = 5.2
POSES = 18
SEED = 1609224499
DERIVATIVE = .0005 * np.eye(5)


class Joystick:
    """Project one physical paddle point onto a single fixed rail."""

    def __init__(self):
        decoder = load_public(FUSED_OUTPUT).decoder
        self.links = decoder.links
        self.point_offset = decoder.offset[3:]+MARKER
        self.translation = decoder.offset[:3]

    def coordinate(self, joint_rows):
        translation, rotations = self.links.read(np.asarray(joint_rows))
        tip = translation + self.translation + rotations@self.point_offset
        return tip@RAIL

    def sensitivity(self,joints):
        side = self.coordinate(np.vstack((joints+DERIVATIVE,joints-DERIVATIVE)))
        return (side[:5]-side[5:])/.001


def plan(joystick, *, count=POSES, seed=SEED):
    rng = np.random.default_rng(seed)
    decoder = load_public(FUSED_OUTPUT).decoder
    poses = []
    while len(poses)<count:
        start = _HOME[:5]+rng.uniform(-1.,1.,5)*[.34,.24,.28,.25,.39]
        body = decoder.body(start)
        if .899 <= body[14] <= 1.00 and np.linalg.norm(body[15:17])<.47:
            poses.append(start)
    seeds = rng.integers(1,2**31-1,count).tolist()
    preflight = np.stack([joystick.sensitivity(q) for q in poses])
    return list(zip(seeds,poses,strict=True)), np.linalg.svd(preflight,compute_uv=False)


class PhysicalTipPaddle:
    """Use native paddle motion only to command the measured physical joystick."""

    def __init__(self,arena,joystick,pose,wiring,spec):
        self.arena, self.joystick, self.wiring, self.spec = arena,joystick,wiring,spec
        arena.reset(seed=SEED,split='practice',render=False)
        arena._prepare_pose(np.r_[pose,-.10])
        self.q = arena._observation(False)['joints'].astype(float)
        self.home = joystick.coordinate(self.q[None])[0]
        self.position = torch.full((1,),.5)
        self.commands,self.traces = [],[self.q.tolist()]
        self.tick = 0

    def step(self,drive):
        if self.tick%2==0:
            gradient = self.joystick.sensitivity(self.q)
            request = MOTOR_LIMIT * float(drive[0]) * gradient/max(.04,np.linalg.norm(gradient))
            applied = np.clip(self.wiring@request,-1.,1.)
            prepare_step(self.arena,applied,PARKED)
            self.q = self.arena._observation(False)['joints'].astype(float)
            y = .5+GAIN*(self.joystick.coordinate(self.q[None])[0]-self.home)
            self.position[0] = np.clip(y,self.spec.paddle_half,1-self.spec.paddle_half)
            self.commands.append(request.tolist())
            self.traces.append(self.q.tolist())
        self.tick+=1


@torch.no_grad()
def point(arena,joystick,seed,pose,wiring,quiet=False,replay=None):
    spec = Physics(opponent_speed=.35)
    game = Pong([seed],spec)
    game.body = paddle = PhysicalTipPaddle(arena,joystick,pose,wiring,spec)
    policy = ScreenPlayer()
    tape, returns = [],0
    for tick in range(NATIVE_CAP):
        if tick%2 == 0:
            if replay is not None:
                command = replay[len(tape)]
            elif quiet:
                command = 0.
            else:
                command = policy.visual(game.render(HEIGHT,WIDTH).numpy()[0],tick,spec)
            tape.append(command)
        outcome = game.step(torch.tensor([command],dtype=torch.float32),auto_reset=False)
        returns+=int(outcome.hits.item())
        score = int(outcome.scores.item())
        if score:
            break
    return dict(seed=seed,pose=pose.tolist(),scalar_commands=tape,commands=paddle.commands,encoders=paddle.traces,
                ticks=tick+1,score=score,returns=returns)


def filter_inputs(clips,alpha,*,shuffled=False):
    output = []
    rng = np.random.default_rng(30864171)
    for clip in clips:
        u = np.cumsum(STEP*np.asarray(clip['commands']),axis=0)
        x = np.empty_like(u)
        prior = np.zeros(5)
        for i,move in enumerate(u):
            prior += alpha*(move-prior)
            x[i] = prior
        y = np.asarray(clip['encoders'])[1:]
        if shuffled:
            y = np.roll(y,int(rng.integers(max(3,len(y)//5),max(4,4*len(y)//5))),axis=0)
        output.append((x-np.mean(x,axis=0),y-np.mean(y,axis=0)))
    return output


def paired_arrays(clips):
    return tuple(np.concatenate(item) for item in zip(*clips,strict=True))


def solve(episodes,regularizer):
    x,y = paired_arrays(episodes)
    return np.linalg.solve(x.T@x+regularizer*I,x.T@y+regularizer*I).T


def error(matrix,episodes):
    x,y = paired_arrays(episodes)
    return float(np.sqrt(np.mean((y-x@matrix.T)**2)))


def fit(plays,*,shuffled=False):
    selected = None
    held = [clip for i,clip in enumerate(plays) if i%4==3]
    train = [clip for i,clip in enumerate(plays) if i%4!=3]
    for alpha in (.13,.29,.54,.77,1.):
        frames = filter_inputs(train,alpha,shuffled=shuffled)
        groups = [frames[::2],frames[1::2]]
        for ridge in (1e-6,.00012,.0017,.016):
            crossval = np.mean([error(solve(a,ridge),b) for a,b in (groups,groups[::-1])])
            if selected is None or crossval<selected['crossval_rms']:
                selected = dict(alpha=alpha,ridge=ridge,crossval_rms=float(crossval))
    frames = filter_inputs(train,selected['alpha'],shuffled=shuffled)
    matrix = solve(frames,selected['ridge'])
    independent = filter_inputs(held,selected['alpha'])
    x,_ = paired_arrays(frames)
    selected.update(matrix=matrix.tolist(),held_encoder_rms=error(matrix,independent),
                    stale_held_encoder_rms=error(I,independent),
                    fit_singular_values=np.linalg.svd(x,compute_uv=False).tolist(),
                    held_source_seeds=[r['seed'] for r in held])
    return selected


def stats(episodes):
    return dict(wins=sum(r['score']>0 for r in episodes),losses=sum(r['score']<0 for r in episodes),
                return_points=sum(r['returns']>0 for r in episodes),total_returns=sum(r['returns'] for r in episodes),
                source_frames=sum(len(r['commands']) for r in episodes))


def source(arena):
    joystick = Joystick()
    scenarios, sensitivity = plan(joystick)
    dense = wirings()[1]
    active,still = [],[]
    for seed,pose in scenarios:
        played = point(arena,joystick,seed,pose,dense)
        constant = point(arena,joystick,seed,pose,dense,quiet=True)
        raw = point(arena,joystick,seed,pose,dense,replay=played['scalar_commands'])
        if digest_json(played)!=digest_json(raw):
            raise ValueError('ordinary tip joystick replay changed the physical plant or point')
        active.append(played)
        still.append(constant)
        print('tip native Pong',seed,played['score'],played['returns'],constant['returns'],flush=True)
    matched,shuffled = fit(active),fit(active,shuffled=True)
    for item in (matched,shuffled):
        item['evaluator_matrix_relative_error'] = float(np.linalg.norm(np.asarray(item['matrix'])-dense)/
                                                        np.linalg.norm(dense))
    spectrum = np.linalg.svd(np.concatenate([r['commands'] for r in active]),compute_uv=False)
    performance = dict(active=stats(active),stationary=stats(still))
    allowed = (performance['active']['return_points']>performance['stationary']['return_points'] and
               spectrum[-1]/spectrum[0]>.009 and
               matched['held_encoder_rms']<.02 and
               matched['held_encoder_rms']<.60*matched['stale_held_encoder_rms'])
    result = dict(one_rail=RAIL.tolist(),rim_offset=MARKER.tolist(),preflight_singular_values=sensitivity.tolist(),
                  raw_command_singular_values=spectrum.tolist(),passed_source=bool(allowed),game=performance,
                  matched=matched,shuffled=shuffled,active=active,stationary=still,native_arm_replays=len(active),
                  carried_bundle_sha256=_hash(BUNDLE))
    _write(SOURCE,result)
    print('source fit',performance,matched,shuffled,spectrum.tolist(),bool(allowed),flush=True)
    return result


def target(arena,result):
    if not result['passed_source']:
        return
    if result['carried_bundle_sha256']!=_hash(BUNDLE):
        raise ValueError('the frozen wrist actor changed')
    policy = CarriedWristPolicy()
    rosters = np.random.default_rng(412201689).integers(1,2**31-1,12).tolist()
    baseline = _read(ROOT/'pong-arm-source.json.gz')
    maps = dict(stale=I,one_axis=np.asarray(baseline['configurations']['matched']),
                matched=np.asarray(result['matched']['matrix']),shuffled=np.asarray(result['shuffled']['matrix']))
    saved = _read(HELD) if HELD.exists() else dict(source_sha256=_hash(SOURCE),rows=[])
    if saved['source_sha256']!=_hash(SOURCE):
        raise ValueError('target data came from a different source fit')
    actual = wirings()[1]
    for seed in rosters[len(saved['rows']):]:
        trials = {}
        for key,matrix in maps.items():
            played = bounce(arena,policy,seed,actual,matrix)
            raw = bounce(arena,policy,seed,actual,matrix,tape=played['actions'])
            if digest_json(played)!=digest_json(raw):
                raise ValueError('dense wrist replay changed its mechanics or score')
            trials[key] = played
        saved['rows'].append(dict(seed=seed,trials=trials))
        _write(HELD,saved)
        print('tip dense wrist',seed,{name:e['metrics']['longest_consecutive_rebounds'] for name,e in trials.items()},flush=True)
    summary = {}
    for name in maps:
        scores = np.asarray([r['trials'][name]['metrics']['longest_consecutive_rebounds'] for r in saved['rows']])
        summary[name] = dict(mean=float(scores.mean()),at_least_ten=int((scores>=10).sum()),maximum=int(scores.max()))
    _write(HELD,saved|dict(summary=summary,replays=len(maps)*len(saved['rows'])))
    print(summary,flush=True)


def forensics(arena):
    """Inspect previously recorded raw commands outside the calibration path."""
    evidence = _read(SOURCE)
    joystick = Joystick()
    dense = wirings()[1]
    originals = np.asarray(evidence['matched']['matrix'])
    receipts = []
    for clip in evidence['active']:
        PhysicalTipPaddle(arena,joystick,np.asarray(clip['pose']),dense,Physics(opponent_speed=.35))
        limited = np.zeros(5,dtype=int)
        contacts = {}
        for requested, expected in zip(clip['commands'],clip['encoders'][1:],strict=True):
            prepare_step(arena,np.clip(dense@requested,-1.,1.),PARKED)
            control = arena._control[:5]
            limited += ((abs(control-arena._control_low[:5])<1e-9) |
                        (abs(control-arena._control_high[:5])<1e-9))
            present = {tuple(sorted(arena._model.geom(int(g)).name for g in contact.geom))
                       for contact in arena._data.contact if contact.efc_address>=0}
            for pair in sorted(present):
                label = ' / '.join(pair)
                contacts[label] = contacts.get(label,0)+1
            if not np.array_equal(arena._observation(False)['joints'],expected):
                raise ValueError('forensic replay changed physical encoders')
        frames=filter_inputs([clip],evidence['matched']['alpha'])[0]
        x,y=frames
        rms = np.sqrt(np.mean((y-x@originals.T)**2,axis=0))
        receipts.append(dict(seed=clip['seed'],frames=len(clip['commands']),limits=limited.tolist(),
                             contact_frames=contacts,encoder_rms_per_joint=rms.tolist()))
    report = dict(after_fit=True, source_sha256=_hash(SOURCE),
                  relative_error_per_row=(np.linalg.norm(originals-dense,axis=1)/
                                          np.linalg.norm(dense,axis=1)).tolist(),
                  total_limit_frames=np.sum([case['limits'] for case in receipts],axis=0).tolist(),
                  source_frames=sum(case['frames'] for case in receipts),cases=receipts)
    _write(FORENSICS,report)
    print('tip diagnostic',report['relative_error_per_row'],report['total_limit_frames'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--held',action='store_true')
    modes.add_argument('--forensics',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    arena = BounceArena(view='wrist',width=192,height=144,max_steps=300)
    try:
        if args.forensics:
            forensics(arena)
            return
        if args.held:
            result = _read(SOURCE)
            last = result['active'][-1]
            raw = point(arena,Joystick(),last['seed'],np.asarray(last['pose']),wirings()[1],
                        replay=last['scalar_commands'])
            if digest_json(last)!=digest_json(raw):
                raise ValueError('last native source tape changed before dense wrist tests')
        else:
            result = source(arena)
        target(arena,result)
    finally:
        arena.close()


if __name__=='__main__':
    main()
