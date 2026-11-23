"""Assign two equally styled wrist movers to a single physical donor head."""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .tt24_bounce import BounceArena, BALL_RADIUS
from .tt24_colour_carry import _angular_distance, _chroma, _ellipse_from_mask
from .tt24_donor_assess import _impact_speeds
from .tt24_donor_trial import Native
from .tt24_object_transfer import CONTACT_RISE, MODEL, OUT, PLAN, WINDOW, encounter
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt4_transfer_trial import _hash, _read, _write

DONOR=OUT/'tt24_donor-refined.npz'
YOKES=OUT/'tt24_donor-extended.json.gz'
GHOST=OUT/'object-transfer-yokes.npz'
RECEIPT=OUT/'object-transfer-held.json.gz'
HELD=tuple(range(370101,370113))
HORIZON=300
WARM=17
PALETTE=np.array(((.76,.065,.88,1.),(.063,.91,.30,1.)))
MODES=('cold','shuffled','pong','geometry','oracle','one_ball')
WIDTH,HEIGHT=192,144


class TwoBallBounce(BounceArena):
    """Render a second sphere along a physical replay with no collision or reward."""

    @staticmethod
    def _document(width,height,detail,arm):
        doc=BounceArena._document(width,height,detail,arm)
        ET.SubElement(doc.find('worldbody'),'geom',name='visual_yoked_sphere',type='sphere',
                      size=str(BALL_RADIUS),pos='0 0 3',rgba='1 1 1 1',contype='0',conaffinity='0')
        return doc

    def __init__(self):
        self.path=None
        self.scene_tick=0
        super().__init__(view='wrist',width=WIDTH,height=HEIGHT,max_steps=HORIZON)
        self._ghost=self._model.geom('visual_yoked_sphere').id

    def reset_fixture(self,seed,path,palette,side):
        self.path=None
        self.scene_tick=0
        self._model.geom_rgba[self._ball_geom]=PALETTE[palette]
        self._model.geom_rgba[self._ghost]=PALETTE[1-palette]
        obs=self.reset(seed=seed,split='held',render=False)
        if path is not None:
            center=self._data.geom_xpos[self._blade].copy()
            self.path=path+center+np.array((side*.047,.012*side,0.))
        obs['rgb']=self.render_wrist()
        return obs

    def step(self,action,**kwargs):
        self.scene_tick+=1
        return super().step(action,**kwargs)

    def _render(self,view):
        self._model.geom_pos[self._ghost]=(0.,0.,3.) if self.path is None else self.path[min(self.scene_tick,len(self.path)-1)]
        return super()._render(view)


def build_yokes(path):
    raw=_read(YOKES)
    data={}
    arena=BounceArena(view='none',max_steps=900)
    try:
        for row in raw['long_stock']['cases']:
            case=row['conditions']['public']
            if len(case['actions'])<HORIZON:
                continue
            arena.reset(seed=row['seed'],split='held',render=False)
            home=arena._data.geom_xpos[arena._blade].copy()
            trajectory=[]
            for action in case['actions'][:HORIZON]:
                trajectory.append(arena._ball_position.copy()-home)
                arena.step(action,render=False)
            data[str(row['seed'])]=np.asarray(trajectory)
    finally:
        arena.close()
    if len(data)<6:
        raise ValueError('need several independent full stock physical yoke tracks')
    np.savez_compressed(path,**data)
    return list(map(int,data))


def schedule(yokes):
    rng=np.random.default_rng(41925611)
    colors=np.tile((0,1),len(HELD)//2)
    rng.shuffle(colors)
    directions=rng.choice((-1,1),len(HELD))
    yoke=np.resize(np.asarray(yokes),len(HELD))
    rng.shuffle(yoke)
    return [dict(seed=seed,yoke=int(slot),palette=int(palette),side=int(side))
            for seed,slot,palette,side in zip(HELD,yoke,colors,directions,strict=True)]


def propose(rgb):
    hue,present=_chroma(rgb)
    centers,_=_chroma(np.rint(255*PALETTE[:,:3]).astype(np.uint8))
    return [_ellipse_from_mask(present&(_angular_distance(hue,region)<.29)) for region in centers]


class Selector:
    """Keep candidate histories and score controlled-object interaction evidence."""

    def __init__(self,mode,model,decoder):
        self.mode=mode
        self.selected=0
        self.model=model
        self.decoder=decoder
        self.proposals=[[],[]]
        self.blades=[]
        self.evidence=np.zeros(2)
        self.quality=np.zeros(2)
        self.last_points=[None,None]
        self.last_change=[None,None]
        self.events=[0,0]

    def read(self,proposals,joints):
        body=self.decoder.body(joints)
        nz=np.sqrt(max(.025,1.-body[15:17]@body[15:17]))
        normal=np.r_[body[15:17],nz]
        normal/=np.linalg.norm(normal)
        self.blades=(self.blades+[body[12:15]])[-WINDOW:]
        self.evidence*=.987
        for i,ellipse in enumerate(proposals):
            position=self.decoder.ball(ellipse,body)
            present=position is not None
            raw=position if present else np.full(3,np.nan)
            self.proposals[i]=(self.proposals[i]+[raw])[-WINDOW:]
            support=0.
            if present:
                support=.85
                if self.last_points[i] is not None:
                    change=position-self.last_points[i]
                    support+=.10*np.exp(-np.linalg.norm(change)/.085)
                    if self.last_change[i] is not None:
                        support+=.05*np.exp(-np.linalg.norm(change-self.last_change[i])/.048)
                    self.last_change[i]=change
                self.last_points[i]=position
            self.quality[i]=.94*self.quality[i]+.06*support
            if self.mode not in ('pong','shuffled','geometry') or len(self.blades)<WINDOW:
                continue
            features=encounter(np.asarray(self.proposals[i]),np.asarray(self.blades),normal,.074,self.decoder.radius)
            if features is None:
                continue
            self.events[i]+=1
            if self.mode=='geometry':
                hit=features[0]*features[2]*features[3]
                vote=np.clip(hit-.35,-.45,1.7)
            else:
                inputs=np.r_[1.,np.clip((features-self.model['center'])/self.model['scale'],-5.,5.)]
                vote=np.clip(inputs@self.model[self.mode],-2.,2.)*.56
            self.evidence[i]=np.clip(self.evidence[i]+vote,-8.,8.)
        score=.40*self.quality+self.evidence
        other=1-self.selected
        if score[other]>score[self.selected]+.040:
            self.selected=other
        return self.selected


def play(arena,mode,setting,path,model,native,weight):
    seed=setting['seed']
    correct=setting['palette']
    obs=arena.reset_fixture(seed,None if mode=='one_ball' else path,correct,setting['side'])
    eyes=[load_public(FUSED_OUTPUT) for _ in range(2)]
    selector=Selector(mode,model,eyes[0].decoder)
    actions,routes=[],[]
    issued,command=np.zeros(5),np.zeros(5)
    while True:
        views=propose(obs['rgb'])
        route=selector.read(views,obs['joints'])
        if mode in ('oracle','one_ball'):
            route=correct
        estimates=[eye.read(ellipse,obs['joints']) for eye,ellipse in zip(eyes,views,strict=True)]
        if len(actions)>=WARM:
            x=np.ascontiguousarray(estimates[route],dtype=float)
            native(x.ctypes.data,weight.ctypes.data,issued.ctypes.data,command.ctypes.data)
        routes.append([int(route==correct),int(views[correct] is not None),int(views[1-correct] is not None)])
        actions.append(command.tolist())
        obs,_,done=arena.step(command)
        if done:
            break
    judge=arena.judge_metrics()
    stock=BounceArena(view='none',max_steps=HORIZON)
    try:
        impact=_impact_speeds(stock,dict(metrics=judge,actions=actions))
    finally:
        stock.close()
    return dict(seed=seed,judge=judge,actions=actions,routes=routes,candidate_events=selector.events,
                power=impact)


def evaluate(args):
    if args.collect_yokes:
        ids=build_yokes(args.yokes)
    else:
        with np.load(args.yokes,allow_pickle=False) as data:
            ids=list(map(int,data.files))
    planned=schedule(ids)
    plan=dict(schema='tt24-object-relevance-test-1',held=planned,horizon=HORIZON,shared_zero_command_ticks=WARM,
              support='two supplied chromatic RGB ellipse proposals, role-balanced palette; public metric reconstruction',
              fixed_scoring='actual top-face rebounds and contact point upward speed >0.055 m/s',
              modes=list(MODES),distractor='noncolliding same-radius sphere replaying independently issued stock drop',
              model_calibration_sha256=_hash(FUSED_OUTPUT),motor_sha256=_hash(args.motor),yokes_sha256=_hash(args.yokes),
              results_are_replay_checked=True)
    if not PLAN.exists():
        _write(PLAN,plan)
    elif _read(PLAN)!=plan:
        raise ValueError('held plan differs from recorded configuration')
    if args.plan_only:
        return
    out={**plan,'source_weights_sha256':_hash(args.model),'episodes':{}}
    if args.resume and args.out.exists():
        old=_read(args.out)
        if any(old[key]!=value for key,value in out.items() if key!='episodes'):
            raise ValueError('existing receipt uses different source or target settings')
        out['episodes']=old['episodes']
    with np.load(args.model,allow_pickle=False) as data:
        model={k:v.copy() for k,v in data.items()}
    with np.load(args.motor,allow_pickle=False) as checkpoint:
        weights=np.ascontiguousarray(checkpoint['weights'],dtype=float)
    with np.load(args.yokes,allow_pickle=False) as data:
        paths={int(k):v.copy() for k,v in data.items()}
    native=Native()
    arena=TwoBallBounce()
    try:
        for mode in args.modes:
            rows=out['episodes'].get(mode,[])
            for setting in planned[len(rows):]:
                row=play(arena,mode,setting,paths[setting['yoke']],model,native.so.tt24_donor_command,weights)
                rows.append(row)
                out['episodes'][mode]=rows
                _write(args.out,out)
                powered=sum(v['velocity_m_s'][2]>CONTACT_RISE for v in row['power'])
                print(mode,row['seed'],row['judge']['rebounds'],powered,round(np.mean(np.asarray(row['routes'])[:,0]),2),flush=True)
    finally:
        arena.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--motor',type=Path,default=DONOR)
    parser.add_argument('--out',type=Path,default=RECEIPT)
    parser.add_argument('--yokes',type=Path,default=GHOST)
    parser.add_argument('--collect-yokes',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    evaluate(parser.parse_args())
