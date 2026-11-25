"""Joint mount optimization using a continuous archive of actual incoming balls."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from . import tt63_fixtures as fixture
from . import tt63_fixture_qualification as q
from . import tt72_retarget_trajectory as pose
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import guarded, verify_clearance
from .tt37_opponent import draw
from .tt39_carry_joints import Carried


ROOT=Path('artifacts/tt84_installation_system')
PLAN=ROOT/'protocol.json'
SCREEN=ROOT/'screen.json'
BUILD=ROOT/'build.json'
OUT=ROOT/'physical.json.gz'
GAMES=ROOT/'games.json.gz'
MP=fixture.mpc
TRAIN=fixture.DEVELOPMENT
TEST=(fixture.RESERVED, fixture.season.ROOT/'after96.json.gz')
DEPTH=np.array([-1.20,-1.34,-1.48,-1.60])
ZERO=np.zeros(5)
DECK='tt29_clamp_tt84_wedge'
LOW=.025
JMAX=4.16
HALF=np.array([.083,.049])
PLATE_OFFSET=.023
FOOT_DEPTH=.0024
RELIEF=.002


def dataset(doc, first=False):
    for i,(c,r) in enumerate(zip(doc['cards'],doc['rows'])):
        if c['profile']!='reach' or any(v['draw']!='random' for v in r['placements']['requested']):continue
        legs=r['placements']['delivered'][:1] if first else r['placements']['delivered']
        for j,o in enumerate(legs):
            if not o.get('receive_line') or not o.get('near_bounce') or not o.get('far_contact'):continue
            f=o['receive_line']; x,y,z=f['xyz']; h=fixture.projected_height(f)
            if h<.88 or h>1.20 or abs(x)>.43:continue
            yield dict(i=i,leg=j,id=c['id'],x=x,height=h,p=f['xyz'],velocity=f['velocity'],
                       stock_landed=bool(o['near_landed']),first_bounce=o['near_bounce']['time'],
                       air_s=o['near_bounce']['time']-o['far_contact']['time'],
                       delivery=digest_json([o['far_contact'],o['near_bounce'],f]))


def freeze():
    d=_read(TRAIN); rng=np.random.default_rng(840712)
    candidates=list(dataset(d)); groups={k:[] for k in ('low','transition','routine')}
    for r in candidates:
        groups['low' if r['height']<1.01 else 'transition' if r['height']<1.048 else 'routine'].append(r)
    archive=[]; raw={}
    for k,items in groups.items():
        raw[k]=len(items)
        for i in rng.permutation(len(items))[:min(len(items),72)]:archive.append(dict(items[i],stratum=k))
    cohort=[]
    for p in TEST:
        doc=_read(p)
        cohort.extend(dict(t,source=str(p)) for t in dataset(doc,first=True))
    # Roles and IDs fixed before the geometry search. Minimize repeated
    # releases, prioritize different lanes, and include ordinary scored balls.
    roster=[]; used=set()
    for tag in ('low','routine'):
        pool=[r for r in cohort if (.89<r['height']<1.026 and not r['stock_landed'] if tag=='low' else
                                   r['stock_landed'] and 1.038<=r['height']<1.17)]
        for sign in (-1,1,0,-1,1):
            eligible=[r for r in pool if r['id'] not in used and r not in roster]
            if not eligible:break
            m=min(eligible,key=lambda r:(abs(r['x']-.105*sign),abs(r['height']-(.994 if tag=='low' else 1.079))))
            roster.append(dict(m,role=tag));used.add(m['id'])
    if len(roster)<7 or used & {r['id'] for r in archive}:
        raise ValueError('Need separate measured opening releases for physical assay')
    ROOT.mkdir(exist_ok=True,parents=True)
    _write(PLAN,dict(schema='tt84-rigid-installation-1',inputs={str(p):_hash(p) for p in (TRAIN,*TEST,fixture.PLAN)},
          ready=d['ready'],archive=archive,natural_strata=raw,physical=roster,
          grid=dict(n=192,sobol_seed=840499,axes=['x_m','depth_m','pitch_deg','roll_deg','yaw_deg'],
                    bounds=[[-.16,.16],[-.009,.009],[-19,19],[-13,13],[-25,25]]),
          policy='Rigid full clamp/core bracket, stock exact hardware, fixed world and continuous opponent. Aggregate archive samples use realized ordinary randomized ball translations, not synthetic goal settings. Shape/velocity proxy only ranks installations. Full native trials must preserve incoming far contact and bounce for physical credit. No served score, trial result or true ball enters the RGB game controller.'))


def contract():
    p=_read(PLAN)
    if any(_hash(Path(f))!=h for f,h in p['inputs'].items()):raise ValueError('Changed archived physical course')
    return p


def mounting(v):
    return Rotation.from_euler('xyz',np.array(v)[2:],degrees=True).as_matrix()


def layout(v, origin, foot):
    x,y,pitch,roll,yaw=v; R=mounting(v); normal=R[:,2]
    anchor=np.array((origin[0]+x,origin[1]+y,origin[2]-FOOT_DEPTH))
    plate=np.array([origin[0]+x,origin[1]+PLATE_OFFSET]); corners=plate+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*HALF
    heights=-(corners-anchor[:2])@normal[:2]/normal[2]
    rise=.005-min(heights); peak=rise+max(heights)
    base=anchor+np.array([0.,0.,rise+FOOT_DEPTH/normal[2]])
    transformed=(foot-origin)@R.T+base
    margin=np.min(HALF-abs(transformed[:,:2]-plate))
    if margin<.00125 or peak>.063 or abs(x)+.087>.705:return None
    return dict(base=base.tolist(),rotation=R.tolist(),peak_m=float(peak),margin_m=float(margin),
                vertices=np.c_[corners,np.full(4,anchor[2]+RELIEF)].tolist()+np.c_[corners,anchor[2]+rise+heights].tolist())


def stock_foot(a):
    m,d=a.model,a.data; b=m.body('learner_base').id
    i=next(i for i in range(m.ngeom) if m.geom_bodyid[i]==b and m.mesh(m.geom_dataid[i]).name=='base_so101_v2')
    k=m.geom_dataid[i]; v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]
    xyz=v@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
    return d.xpos[b].copy(),xyz[np.abs(xyz[:,2]-(d.xpos[b,2]-FOOT_DEPTH))<.00015]


def windows(archive):
    owner=[]; Ps=[]; Vs=[]; Ns=[]; Us=[]; Target=[]; lead=[]
    for j,r in enumerate(archive):
        p=np.array(r['p']); v=np.array(r['velocity']); oy=np.array(DEPTH)-p[1]; times=(oy/max(1.,abs(v[1])*.97))*-1
        for depth,t in zip(DEPTH,times,strict=True):
            pc=p+v*t+np.array([0.,0.,-4.905*t*t]); pc[1]=depth; vc=v+[0.,0.,-9.81*t]
            if pc[2]<.794:continue
            for pace in (3.65,4.23):
                late=(.70-depth)/(.91*pace); mid=-depth/(.945*pace)
                up=max((.78-pc[2])/late+4.905*late,(.983-pc[2])/mid+4.905*mid)
                out=np.array([-.70*pc[0]/late,pace,up]); n=out-vc; n/=np.linalg.norm(n)
                vertical=np.array([0.,0.,1.])-n*n[2]; vertical/=np.linalg.norm(vertical)
                # Directional minimums combine normal impulse and active tangential lift.
                target=[np.clip((out@n+.83*(vc@n))/1.83,.055,.95),.25+.14*max(0.,up)]
                owner.append(j);Ps.append(pc);Vs.append(vc);Ns.append(n);Us.append(vertical);Target.append(target)
                lead.append(float(r['air_s'] + max(.035,(depth+0.39)/min(-1.5,v[1])) - .07))
    return tuple(map(np.asarray,(owner,Ps,Vs,Ns,Us,Target,lead)))


class Atlas:
    def __init__(self,a):
        model=pose.FullPose(a); self.origin,self.foot=stock_foot(a)
        stock=a.joints[:5]
        samples=qmc.scale(qmc.Sobol(5,scramble=True,seed=840471).random_base2(14),model.low,model.high)
        z=qmc.scale(qmc.Sobol(5,scramble=True,seed=840472).random_base2(12),np.maximum(model.low,stock-1.25),np.minimum(model.high,stock+1.25))
        self.q=np.vstack([stock,samples,z]); ps=[]; fs=[]; Js=[]; Ws=[]
        for q0 in self.q:
            p,R,J,W=model.frame(q0,True);ps.append(p-self.origin);fs.append(R[:,2].copy());Js.append(J);Ws.append(W)
        self.p=np.array(ps); self.n=np.array(fs);self.J=np.array(Js);self.W=np.array(Ws)
        self.scale=np.array([1.,1.,1.,.126,.126,.126]); base=np.c_[self.p,self.n]; opposite=np.c_[self.p,-self.n]
        self.tree=cKDTree(np.vstack([base,opposite])*self.scale)

    def grade(self,mount,D,tags,goal):
        owner, ball, _, n, vert, ideal, lead=D; R=np.asarray(mount['rotation']); root=np.asarray(mount['base']);
        localP=(ball-root)@R; localN=n@R; localU=vert@R
        queries=np.c_[localP-.023*localN,localN]*self.scale
        _, ix=self.tree.query(queries,k=10); ix%=len(self.p)
        target=np.r_[(goal[:3]-root)@R, goal[3:]@R]*self.scale
        _, init=self.tree.query(target,k=10); init=init%len(self.p)
        travel=np.min(np.max(abs(self.q[ix,None,:]-self.q[init]),axis=3),axis=2)
        residual=localP[:,None,:]-self.p[ix]; d=np.einsum('nki,ni->nk',residual,localN)
        r=np.linalg.norm(residual-d[:,:,None]*localN[:,None,:],axis=2)
        alignment=abs(np.einsum('nki,ni->nk',self.n[ix],localN));
        at=residual-.020*localN[:,None,:]
        J=self.J[ix]+np.cross(self.W[ix].transpose(0,1,3,2),at[:,:,None,:]).transpose(0,1,3,2)
        normal=np.einsum('ni,nkid->nkd',localN,J); lift=np.einsum('ni,nkid->nkd',localU,J)
        # 2-D box support, including mixed directions to avoid using the
        # same saturated joint independently for the two desired speeds.
        vel=np.ones_like(d,dtype=bool); margin=np.full(d.shape,np.inf)
        for phi in np.arange(8)*np.pi/4:
            c,s=np.cos(phi),np.sin(phi)
            m=JMAX*np.abs(c*normal+s*lift).sum(axis=2) - (ideal[:,0]*c+ideal[:,1]*s)[:,None]
            margin=np.minimum(margin,m);vel&=m>=-.005
        feasible=(d>.007)&(d<.045)&(r<.058)&(alignment>.90)&(travel<=lead[:,None]*3.9+.16)
        tri=(d>.002)&(d<.053)&(r<.093)&(alignment>.84)
        value=np.where(tri,1.-7.*r-abs(d-.023)*4.0-.62*(1.-alignment)+.08*np.clip(margin,-.5,.5)-.065*travel,0.)
        n=len(tags); works=np.zeros(n,bool);interior=np.zeros(n,bool);soft=np.zeros(n);np.logical_or.at(works,owner,(feasible&vel).any(axis=1));
        np.logical_or.at(interior,owner,feasible.any(axis=1));np.maximum.at(soft,owner,value.max(axis=1));
        output={}
        for label in ('low','transition','routine'):
            group=tags==label
            output[label]=dict(n=int(group.sum()),pose=int(interior[group].sum()),impulse=int(works[group].sum()),
                               soft=float(soft[group].mean()))
        return output


def screen():
    p=contract(); source=p['archive']; tags=np.array([s['stratum'] for s in source]); D=windows(source)
    row0=dict(v=[0.]*5,kind='stock');lat=dict(v=[.12,0.,0.,0.,0.],kind='lateral12')
    d=dict(protocol=_hash(PLAN),operator=_hash(Path(__file__)),rows=[],complete=False)
    with MP.centered(p['ready']):
        a=MP.arena(width=64,height=48,detail='cad');
        try:
            atlas=Atlas(a); goal0=pose.FullPose(a).frame(np.asarray(p['ready'])[:5]); goal=np.r_[goal0[0],goal0[1][:,2]]
            options=qmc.scale(qmc.Sobol(5,scramble=True,seed=p['grid']['sobol_seed']).random_base2(8)[:p['grid']['n']],
                              [b[0] for b in p['grid']['bounds']],[b[1] for b in p['grid']['bounds']]);
            for i,v in enumerate([row0['v'],lat['v'],*np.round(options,5).tolist()]):
                kind=('stock','lateral12')[i] if i<2 else 'combined'
                if i<2:m=dict(base=(atlas.origin+[v[0],0,0]).tolist(),rotation=np.eye(3).tolist(),peak_m=0.,margin_m=None)
                else:m=layout(v,atlas.origin,atlas.foot)
                if m is None:continue
                score=atlas.grade(m,D,tags,goal); d['rows'].append(dict(v=v,kind=kind,mount=m,stats=score))
            base=d['rows'][0]['stats']
            valid=[r for r in d['rows'] if r['kind']=='combined' and
                   r['stats']['routine']['impulse']>=.92*base['routine']['impulse'] and
                   r['stats']['transition']['impulse']>=.85*base['transition']['impulse']]
            rank=lambda r: (sum(r['stats'][s]['impulse']*w for s,w in [('low',1.),('transition',.8),('routine',.7)]) +
                         sum(r['stats'][s]['soft']*w for s,w in [('low',5.),('transition',3.),('routine',2.)]))
            good=[r for r in valid if rank(r)>rank(d['rows'][0]) and r['stats']['low']['impulse']>base['low']['impulse']]
            shortlist=sorted(good,key=rank,reverse=True)[:6]
            d.update(complete=True,score_stock=rank(d['rows'][0]),score_lateral=rank(d['rows'][1]), shortlist=shortlist,
                     baseline_ready_blade=goal.tolist(),hardware_accepted=len(d['rows'])-2, candidates=len(options))
            _write(SCREEN,d)
            for r in d['rows'][:2]+shortlist:
                print(r['kind'],r['v'],rank(r),{k:v['impulse'] for k,v in r['stats'].items()},flush=True)
        finally:a.close()


@contextmanager
def installed(v, shape):
    if not any(v):
        yield
        return
    old=fixture.TT20Arena._model_document
    def document(a):
        d=old(a); w=d.find('worldbody');base=w.find("body[@name='learner_base']");
        clamp=w.find("body[@name='tt20_table_end_clamp']");clamp.set('pos',fixture._vector([v[0],0.,0.]));
        qb=np.fromstring(base.get('quat'),sep=' ');
        R=Rotation.from_matrix(np.array(shape['rotation']))*Rotation.from_quat(qb[[1,2,3,0]])
        base.set('pos',fixture._vector(shape['base']));base.set('quat',fixture._vector(R.as_quat()[[3,0,1,2]]))
        verts=np.asarray(shape['vertices']).copy();verts[:,0]-=v[0]
        ET.SubElement(d.find('asset'),'mesh',name=DECK,vertex=fixture._vector(verts.ravel()),
                      face='0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7')
        ET.SubElement(clamp,'geom',name=DECK,type='mesh',mesh=DECK,rgba='.20 .23 .26 1',contype='2',conaffinity='5')
        y=np.mean(verts[:4,1]); top=verts[0,2]-RELIEF
        fixture._box(clamp,DECK+'_center_pad',(0.,y-.018,top+.5*RELIEF),(.049,.029,.5*RELIEF),(.22,.25,.29))
        fixture._box(clamp,DECK+'_inboard_pad',(0.,y+.031,top+.5*RELIEF),(.073,.018,.5*RELIEF),(.22,.25,.29))
        return d
    with patch.object(fixture.TT20Arena,'_model_document',document):
        yield


def physical_support(a):
    m,d=a.model,a.data; base=m.body('learner_base').id
    b=[i for i in range(m.ngeom) if m.geom_bodyid[i]==base]; upper=m.geom(DECK).id
    pads=[m.geom(DECK+k).id for k in ('_center_pad','_inboard_pad')]; plate=m.geom('tt29_clamp_top_plate').id
    jaw=[m.geom(f'tt29_clamp_{i}_upper_jaw').id for i in (0,1)]
    dist=lambda i,j:float(mujoco.mj_geomDistance(m,d,i,j,.10,None))
    top=min(dist(i,upper) for i in b); bearings=[dist(i,j) for i in pads for j in (upper,plate)]
    g=min(dist(i,k) for i in [upper,*pads] for k in jaw)
    table=min(dist(i,m.geom(s).id) for i in [upper,*pads,*b] for s in a.table_panels)
    return dict(base_gap_m=top,bearing_m=bearings,jaw_gap_m=g,table_gap_m=table,
                safe=bool(abs(top)<.00016 and max(map(abs,bearings))<.00016 and g>.00075 and table>.001))


def invariants(m):
    g=[i for i in range(m.ngeom) if not m.geom(i).name.startswith(DECK)]
    b=[i for i in range(m.nbody) if m.body(i).name!='tt20_table_end_clamp']
    return ({k:np.copy(getattr(m,k)) for k in ('jnt_range','jnt_axis','actuator_forcerange','actuator_ctrlrange','dof_damping','cam_pos','cam_quat','eq_data')}
       | {k:np.copy(getattr(m,k)[b]) for k in ('body_mass','body_ipos','body_inertia','body_iquat')}
       | {k:np.copy(getattr(m,k)[g]) for k in ('geom_pos','geom_size','geom_quat','geom_friction','geom_contype','geom_conaffinity')})


def build():
    p=contract(); s=_read(SCREEN); res=dict(screen_sha256=_hash(SCREEN),protocol=_hash(PLAN),rows=[],chosen=None)
    if not s['complete'] or s['protocol']!=_hash(PLAN):raise ValueError('Incomplete physical system ranking')
    ready=np.array(p['ready']); goal=np.array(s['baseline_ready_blade']); basis=_read(fixture.grip.PREPARED)['rows']; reference=None; floor=None
    for r in [s['rows'][0],*s['shortlist']]:
        with installed(r['v'],r['mount']),MP.centered(ready):
            a=MP.arena(width=64,height=48,detail='cad')
            try:
                sig=invariants(a.model)
                if reference is None:reference=sig
                if any(not np.array_equal(sig[k],v) for k,v in reference.items()):raise ValueError('Non-installation feature changed')
                nominal=r['kind']=='stock'; safe=fixture.support_check(a,'stock') if nominal else physical_support(a)
                commission=(dict(ready=ready.tolist()) if nominal else fixture.calibrate(a,ready,goal)) if safe['safe'] else None
                result=fixture.grip.scan_arena(a,np.array(commission['ready']),basis) if commission else {}
                count=result.get('ordinary_covered'); res['rows'].append(dict(candidate=r,support=safe,commission=commission,ordinary=count,atlas=result.get('ordinary')))
                if nominal:floor=count
                if not nominal and count is not None and count>=.85*floor:
                    res['chosen']=len(res['rows'])-1;break
            finally:a.close()
    _write(BUILD,res)
    print('strict CAD fixture',res['chosen'],[(r['candidate']['v'],r['support']['safe'],r['ordinary']) for r in res['rows']],flush=True)


def physical():
    p=contract(); b=_read(BUILD)
    if b['chosen'] is None:
        raise ValueError('No whole work-cell installation passed physical support and routine pose routes')
    bodies=[b['rows'][0],b['rows'][b['chosen']]]; docs={}
    result=dict(protocol=_hash(PLAN),build=_hash(BUILD),rows=[],complete=False)
    if OUT.exists():
        old=_read(OUT)
        if any(old[k]!=v for k,v in result.items() if k not in ('rows','complete')):raise ValueError('Changed real fixture comparison')
        result=old
    with fixture.school.enabled(cameras=False):
        schedule=[(c,i) for c in p['physical'] for i in range(2)]
        for spec,i in schedule[len(result['rows']):]:
            shape=bodies[i]['candidate']; ready=bodies[i]['commission']['ready']; source=spec['source'];
            doc=docs[source] if source in docs else docs.setdefault(source,_read(Path(source)))
            with installed(shape['v'],shape['mount']),MP.centered(ready):
                a=MP.arena(width=64,height=48); cad=MP.arena(width=64,height=48,detail='cad')
                try:
                    guard=q.GripClearance(cad.model,stock=True)
                    out=q.score(a,guard,doc,spec,'stock' if i==0 else 'installed',3)
                finally:a.close();cad.close()
            real=any(q.received(t) for t in out['physical']); wide=any(q.received(t,True) for t in out['physical']);
            reference=q.received(out['stock_commands'])
            result['rows'].append(dict(case=spec,body=i,real=real,broad=wide,reference=reference,trial=out))
            result['complete']=len(result['rows'])==len(schedule);_write(OUT,result)
            print('matched',spec['id'],spec['role'],i,out['matched_prefix'],real,wide,reference,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('plan','screen','build','physical'))
    {'plan':freeze,'screen':screen,'build':build,'physical':physical}[p.parse_args().mode]()
