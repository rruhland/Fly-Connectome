"""Separate SO-101 body-transfer experiment with buildable rigid installations."""

import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from . import tt38_joint_mpc as mpc
from . import tt41_shallow_clamp as shim
from . import tt41_shallow_grip as grip
from . import tt46_continuous_season as season
from . import tt57_motor_school as school
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt20_fixture import TABLE_EDGE, TT20Arena, _box, _vector
from .tt37_bodycoach_windows import Reach
from .tt39_opportunity_map import projected_height
from .tt40_low_feasibility import Faces


ROOT = Path('artifacts/tt63_fixtures')
PLAN = ROOT/'protocol.json'
GEOMETRY = ROOT/'geometry.json'
DEVELOPMENT = season.ROOT/'learning.json.gz'
RESERVED = season.ROOT/'after128.json.gz'
DEVELOPMENT_INDEXES = (10, 12, 15, 17, 21, 75, 80)
LOW_HELD = (12,19,23)
NORMAL_HELD = (3,5,14,22)
CONFIGS = {'stock':{}, 'neck-plus':dict(pitch=17.,seat='out-8'),
           'neck-minus':dict(pitch=-17.,seat='out-8'),
           'neck-plus22':dict(pitch=22.,seat='out-16'),
           'neck-minus22':dict(pitch=-22.,seat='out-16'),
           'low-shelf26':dict(drop=.026), 'low-shelf48':dict(drop=.048), 'low-shelf66':dict(drop=.066)}
SHELF = 'tt29_clamp_tt63_deck'
NECK = -.032  # within the stock shaft capsule and stock blade disk at the same time
MASS = .115


def freeze():
    d,h=_read(DEVELOPMENT),_read(RESERVED)
    if d['ready']!=h['ready']:
        raise ValueError('Stock ready differs between phases')
    def cases(src, doc, idx, role):
        result=[]
        for i in idx:
            c,r=doc['cards'][i],doc['rows'][i]
            o=r['placements']['delivered'][0]
            if bool(o['near_landed'])!=(role=='routine'):
                raise ValueError('Wrong frozen first-offer role')
            result.append(dict(i=i,id=c['id'],source=str(src),role=role,x=o['receive_line']['xyz'][0],
                               height=projected_height(o['receive_line']),touched=bool(o.get('near_contact')),
                               delivery=digest_json([o['far_contact'],o['near_bounce'],o['receive_line']])))
        return result
    dev=cases(DEVELOPMENT,d,DEVELOPMENT_INDEXES,'development')
    held=cases(RESERVED,h,LOW_HELD,'low')+cases(RESERVED,h,NORMAL_HELD,'routine')
    if {r['id'] for r in dev}&{r['id'] for r in held}:
        raise ValueError('Leaked fixture-development release into assessment')
    ROOT.mkdir(exist_ok=True,parents=True)
    _write(PLAN,dict(schema='tt63-rigid-fixture-generalization-1', configurations=CONFIGS, ready=d['ready'],
         checks={str(p):_hash(p) for p in (DEVELOPMENT,RESERVED,grip.PREPARED,school.course.OPTIC,mpc.BANK)},
         development=dev, held=held, game_indexes=[0,2,6], cap=204, interior_m=.058, physical_budget=3,
         geometry_gate='Stock-contact-radius improvement >=4 mm on at least two pre-existing developmental rays and >=85% stock regular-height atlas coverage, or at least two real CAD-clear interior development strokes plus the same atlas criterion. Inspect every installed fixture, not just a favorable score.',
         shelf_ready='The unadjusted stock encoder vector intersects the deck or table on a different mount. Use the nearest CAD-clear blade center/axis to standard stock ready on a preassigned geometric grid. This is commissioning per body, not performance selection. Test the stock encoder tuple as a separate static control.',
         scope='Isolated body generalization. Blade pitch is perpendicular to shaft at a point embedded in both real shapes while the stock shaft still seats in its jaws. Paddles retain individual masses; their spatial inertias are recompiled. The outside drop shelf bolts to the same steel clamp spines, keeps the stock SO-101 and paddle intact, and lowers the deck without passing through the tabletop. Native ball, table, laws and partner are stock. Native coaching is diagnostic only. Live policy, if warranted, is separately RGB and encoder-only.'))


def neck_geometry(element, degrees):
    q=np.fromstring(element.get('quat'),sep=' ')
    p=np.fromstring(element.get('pos'),sep=' ')
    rot=Rotation.from_quat(q[[1,2,3,0]])
    v=rot.apply([0.,0.,1.]); shaft=np.array([0.,0.,1.]); axis=np.cross(shaft,v)
    if abs(v@shaft)>1e-6 or np.linalg.norm(axis)<.99:
        raise ValueError('Expected stock perpendicular blade face')
    pivot=shaft*NECK
    turn=Rotation.from_rotvec(np.deg2rad(degrees)*axis)
    center=pivot+turn.apply(p-pivot)
    nominal=np.fromstring(element.get('size'),sep=' ')
    near=pivot-center; after=turn.apply(v)
    if np.linalg.norm(near-after*(near@after))>=nominal[0]-.010 or abs(near@after)>.00001:
        raise ValueError('Neck pivot no longer forms an embedded physical joint')
    return center,(turn*rot).as_quat()[[3,0,1,2]]


@contextmanager
def mounted(name):
    opts=CONFIGS[name]
    if not opts:
        yield
        return
    original=TT20Arena._model_document
    def document(self):
        d=original(self)
        if self.fixture!=TABLE_EDGE:
            raise ValueError('Only regulation end-clamp benchmark is calibrated')
        w=d.find('worldbody')
        if 'pitch' in opts:
            blade=w.find("body[@name='learner_paddle']/geom[@name='learner_racket']")
            p,q=neck_geometry(blade,opts['pitch']); blade.set('pos',_vector(p)); blade.set('quat',_vector(q))
        else:
            end=-self.plant.table_half_length
            c=w.find("body[@name='tt20_table_end_clamp']")
            base=w.find("body[@name='learner_base']")
            old=np.fromstring(base.get('pos'),sep=' ')
            plate=c.find("geom[@name='tt29_clamp_top_plate']")
            top=np.fromstring(plate.get('pos'),sep=' ')[2]+np.fromstring(plate.get('size'),sep=' ')[2]
            top-=opts['drop']
            base.set('pos',_vector((0.,end-.072,top+.0024)))
            # Wings overlap the two original exterior spines, not the table.
            # The lowered steel deck is integral to the wings. No dynamic link
            # or unmodeled robot support is added.
            _box(c,SHELF,(0.,end-.055,top-.003),(.084,.051,.003),(.19,.23,.28))
            for j,x in enumerate((-.068,.068)):
                _box(c,f'tt29_clamp_tt63_web_{j}',(x,end-.052,top-.013),(.0115,.041,.010),(.24,.28,.30))
        return d
    with ExitStack() as stack:
        if 'pitch' in opts:
            stack.enter_context(grip.mounted(opts['seat']))
        stack.enter_context(patch.object(TT20Arena,'_model_document',document))
        yield


def identity(model, name):
    """Exclude only the articulated installation's explicitly reported changes."""
    g=[i for i in range(model.ngeom) if not model.geom(i).name.startswith('tt29_clamp_tt63_')]
    geom_body=[model.geom_bodyid[i] for i in g]
    b0=model.body('learner_base').id
    paddle=model.body('learner_paddle').id
    clamp=model.body('tt20_table_end_clamp').id
    b=[i for i in range(model.nbody) if i not in (b0,paddle,clamp)]
    e=[i for i in range(model.neq) if i!=model.equality('tt19_grasp').id]
    fixed=[i for i in g if i!=model.geom('learner_racket').id]
    return ({k:np.copy(getattr(model,k)) for k in ('jnt_axis','jnt_range','jnt_pos','dof_damping',
             'actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_trnid','cam_pos','cam_quat')}
       | {k:np.copy(getattr(model,k)[b]) for k in ('body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat')}
       | {k:np.copy(getattr(model,k)[g]) for k in ('geom_size','geom_type','geom_contype','geom_conaffinity','geom_friction')}
       | {k:np.copy(getattr(model,k)[fixed]) for k in ('geom_pos','geom_quat')}
       | {f'base_{k}':np.copy(getattr(model,k)[b0]) for k in ('body_quat','body_mass','body_inertia','body_ipos','body_iquat')}
       | dict(eq=np.copy(model.eq_data[e]), geom_body=np.asarray(geom_body)))


def support_check(a,name):
    opt=CONFIGS[name]; m,d=a.model,a.data
    if 'pitch' not in opt and 'drop' not in opt:
        return shim.static_check(a,0.)
    if 'pitch' in opt:
        r=grip.hold_check(a,opt['seat'])
        r['safe']=bool(r['bilateral_span_m']>=.005-1e-8 and r['least_blade_to_wrist_m']>.001)
        return r
    base=m.body('learner_base').id; fixed=m.body('tt20_table_end_clamp').id
    bgeoms=np.flatnonzero(m.geom_bodyid==base)
    name_set=[m.geom(i).name for i in range(m.ngeom)]
    sids=[m.geom(n).id for n in a.table_panels]+[i for i,n in enumerate(name_set) if m.geom_bodyid[i]==fixed]
    q=[dict(surface=m.geom(j).name,base=m.mesh(m.geom_dataid[i]).name,
             gap_m=float(mujoco.mj_geomDistance(m,d,i,j,.15,None))) for i in bgeoms for j in sids]
    gap=min(x['gap_m'] for x in q if x['surface']==SHELF)
    deck=m.geom(SHELF); deck_plane=d.geom_xpos[deck.id,2]+deck.size[2]
    deck_min=d.geom_xpos[deck.id]-deck.size; deck_max=d.geom_xpos[deck.id]+deck.size
    mesh=m.geom(bgeoms[1]); bpos=d.geom_xpos[mesh.id]; R=d.geom_xmat[mesh.id].reshape(3,3)
    data=mesh.dataid[0]; points=m.mesh_vert[m.mesh_vertadr[data]:m.mesh_vertadr[data]+m.mesh_vertnum[data]]@R.T+bpos
    foot=points[np.abs(points[:,2]-deck_plane)<.00015,:2]
    foot_margin=float(np.min(np.minimum(foot-deck_min[:2],deck_max[:2]-foot))) if len(foot) else -1.
    min_table=min(float(mujoco.mj_geomDistance(m,d,i,m.geom(n).id,.15,None)) for i,v in enumerate(name_set)
                  if v.startswith('tt29_clamp_tt63_') for n in a.table_panels)
    return dict(base_contact_m=gap,base_footprint_margin_m=foot_margin,deck_table_gap_m=min_table,
                safe=bool(abs(gap)<.00015 and foot_margin>.001 and min_table>.001
                          and min(x['gap_m'] for x in q)>-.00015),pairs=q)


def check():
    p=_read(PLAN)
    if p['configurations']!=CONFIGS or any(_hash(Path(f))!=h for f,h in p['checks'].items()):
        raise ValueError('Frozen physical fixture docket changed')
    return p


def calibrate(a, ready, target):
    """Commission a new clamp with a sensor-geometric, outcome-blind ready pose."""
    kin=Faces(a); guard=Reach(a,ready,ready)
    if not np.isfinite(target).all():
        raise ValueError('Invalid surveyed stock ready target')
    p,n=target[:3],target[3:]; both=(n,-n)
    raw=qmc.scale(qmc.Sobol(5,scramble=True,seed=630413).random_base2(12),kin.low,kin.high)
    poses=np.array([np.r_[*kin.pose(q)] for q in raw]); output=[]
    for normal in both:
        score=np.linalg.norm(poses[:,:3]-p,axis=1)+.12*np.linalg.norm(poses[:,3:]-normal,axis=1)
        for j in np.argsort(score)[:40]:
            for h in (0.,.025,.05):
                q,(pos,axis,*_)=kin.fit(p+[0.,0.,h],normal,raw[j]);
                if (not guard.clear(q) or pos[2]<.88 or abs(pos[0])>.18
                        or axis@normal<.88 or np.linalg.norm(pos-p)>.095):
                    continue
                error=np.linalg.norm(pos-p)+.12*(1-axis@normal)+.006*np.linalg.norm(q-ready[:5])
                output.append((error,np.r_[q,ready[5]],pos,axis))
    if not output:return None
    cost,q,p,n=min(output,key=lambda x:x[0])
    return dict(ready=q.tolist(),center=p.tolist(),axis=n.tolist(),loss=float(cost),
                stock_ready_clear=bool(guard.clear(ready[:5])))


def geometry():
    p=check();ready=np.array(p['ready']);source=_read(grip.PREPARED)['rows']; rows=[]
    r=dict(protocol=_hash(PLAN),geometry_code=_hash(Path(__file__)),rows=rows,complete=False)
    baseline=None; goal=None
    for name in CONFIGS:
        with mounted(name),mpc.centered(ready):
            a=mpc.arena(width=64,height=48,detail='cad')
            try:
                ident=identity(a.model,name)
                if baseline is None:
                    baseline=ident
                    goal=np.r_[*Faces(a).pose(ready[:5])]
                if any(not np.array_equal(ident[k],v) for k,v in baseline.items()):
                    raise ValueError('An unrelated arm, opponent, table, camera or controller property changed')
                mass=float(a.model.body('learner_paddle').mass[0]); i=a.model.body('learner_paddle').id
                if abs(mass-MASS)>1e-9:
                    raise ValueError('Paddle mass or material was changed')
                support=support_check(a,name)
                row=dict(name=name,static=support,body_base=a.model.body('learner_base').pos.tolist(),
                         paddle_mass_kg=mass,paddle_diagonal_inertia=a.model.body_inertia[i].tolist(),
                         paddle_inertial_center=a.model.body_ipos[i].tolist(),
                         blade_position=a.model.geom('learner_racket').pos.tolist(),
                         blade_orientation=a.model.geom('learner_racket').quat.tolist())
                if support['safe']:
                    if 'drop' in CONFIGS[name]:
                        commission=calibrate(a,ready,goal)
                        row['commission']=commission
                        if commission is not None:
                            prep=np.array(commission['ready'])
                            local=[dict(r,prepared=prep[:5].tolist()) for r in source]
                            row.update(grip.scan_arena(a,prep,local))
                    else:
                        row.update(grip.scan_arena(a,ready,source))
                if rows and 'ordinary' in row:
                    row.update(grip.compare(rows[0],row))
                rows.append(row); _write(GEOMETRY,r)
                print(name,support['safe'],row.get('reason'),row.get('ordinary_covered'),row.get('live_eligible'),
                      [round(1000*v['best_m']) if v['best_m'] is not None else None for v in row.get('low',[])],flush=True)
            finally:a.close()
    r['complete']=True
    _write(GEOMETRY,r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','geometry'))
    a=p.parse_args()
    freeze() if a.mode=='plan' else geometry()
