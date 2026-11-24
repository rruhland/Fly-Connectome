"""Separate pitched adapter on the original edge clamp, with no plant retuning."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .tt20_fixture import TT20Arena, TABLE_EDGE, _vector
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit as score_events
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt39_opportunity_map import projected_height
from .tt41_shallow_grip import (
    BANK, OPTS, PREPARED, ROOT, compare, physical_live, scan_arena,
)
from .tt4_transfer_trial import _hash, _read, _write


ANGLES=(0.,4.,6.,-4.)
XHALF=.061
YLIM=(-.025,.066)
MIN_WEDGE=.0015
ADAPTER='tt41_clamp_pitched_spacer'


def parameters(degrees):
    slope=float(np.tan(np.deg2rad(degrees)))
    raise_m=MIN_WEDGE-min(y*slope for y in YLIM) if degrees else 0.
    return dict(degrees=degrees,thickness_center_m=raise_m,top_min_m=MIN_WEDGE if degrees else 0.,
                thickness_pad_end_m=[raise_m+y*slope for y in YLIM])


@contextmanager
def pitched(degrees):
    if degrees==0.:
        yield
        return
    if degrees not in ANGLES:raise ValueError('Adapter pitch was not predeclared')
    original=TT20Arena._model_document

    def document(self):
        doc=original(self)
        if self.fixture!=TABLE_EDGE:raise ValueError('Adapter only fastens to the published top clamp plate')
        world=doc.find('worldbody');base=world.find("body[@name='learner_base']")
        support=world.find("body[@name='tt20_table_end_clamp']")
        plate=support.find("geom[@name='tt29_clamp_top_plate']")
        position=np.fromstring(base.get('pos'),sep=' ');q=np.fromstring(base.get('quat'),sep=' ')
        top=np.fromstring(plate.get('pos'),sep=' ')+np.fromstring(plate.get('size'),sep=' ')
        slope=float(np.tan(np.deg2rad(degrees)));rise=parameters(degrees)['thickness_center_m']
        vertices=[(x,position[1]+y,top[2]+(rise+slope*y if upper else 0.))
                  for upper in (False,True) for x,y in [(-XHALF,YLIM[0]),(XHALF,YLIM[0]),
                                                   (XHALF,YLIM[1]),(-XHALF,YLIM[1])]]
        if XHALF>top[0] or np.min(np.array(vertices)[:,1])<top[1]-.10 or np.max(np.array(vertices)[:,1])>top[1]:
            raise ValueError('Adapter protrudes from the existing adhered clamp plate')
        ET.SubElement(doc.find('asset'),'mesh',name=ADAPTER,vertex=_vector(np.ravel(vertices)),
                      face='0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7')
        ET.SubElement(support,'geom',name=ADAPTER,type='mesh',mesh=ADAPTER,
                      rgba='.17 .20 .23 1',contype='2',conaffinity='5')
        r=Rotation.from_rotvec([np.deg2rad(degrees),0.,0.])*Rotation.from_quat(q[[1,2,3,0]])
        base.set('quat',_vector(r.as_quat()[[3,0,1,2]]))
        base.set('pos',_vector(position+[0.,0.,rise]))
        return doc

    with patch.object(TT20Arena,'_model_document',document):
        yield


def plan(grip_file,out):
    parent=_read(grip_file)
    _write(out,dict(adapter=[parameters(k) for k in ANGLES],half_width_m=XHALF,depth_offsets_m=YLIM,
          parent_sha256=_hash(grip_file),ready=parent['ready'],cards=parent['cards'],cap=parent['cap'],
          invariants='Original SO-101 (body CAD, mass, range, servo, paddle and weld), same plate, table, far player and draws. '
          'Only static fastened spacer and base origin/angle change.',
          selection='Same +4mm in two exact historical low trajectories and ≥85% normal-height coverage gate as the fixed grip, '
          'with no base/wedge or base/court CAD penetration beyond the nominal contact convention.'))


def static_check(a,degrees):
    m,d=a.model,a.data;bid=m.body('learner_base').id
    body=[i for i in range(m.ngeom) if m.geom_bodyid[i]==bid]
    surfaces=[m.geom(n).id for n in (*a.table_panels,'tt29_clamp_top_plate')]
    if degrees!=0.:surfaces.append(m.geom(ADAPTER).id)
    pairs=[dict(surface=m.geom(j).name,base_mesh=m.mesh(m.geom_dataid[i]).name,
                gap_m=float(mujoco.mj_geomDistance(m,d,i,j,.15,None))) for i in body for j in surfaces]
    supported=[p['gap_m'] for p in pairs if p['surface']==(ADAPTER if degrees else 'tt29_clamp_top_plate')]
    return dict(pairs=pairs,base_contact_m=min(supported),safe=(min(p['gap_m'] for p in pairs)>-.00015 and
                       -.00015<min(supported)<.00015))


def geometry(procedure,out):
    p=_read(procedure);ready=np.array(p['ready'])
    if p['adapter']!=[parameters(k) for k in ANGLES]:raise ValueError('Frozen adapter dimensions changed')
    data=_read(PREPARED)['rows'];rows=[];baseline=None
    for angle in ANGLES:
        with pitched(angle),centered(ready):
            a=arena(detail='cad')
            try:
                current=unchanged(a.model,angle)
                if baseline is None:baseline=current
                if any(not np.array_equal(current[k],v) for k,v in baseline.items()):
                    raise ValueError('Original SO-101, motor, ball or playing surfaces changed')
                physical=static_check(a,angle)
                row=dict(angle=angle,static=physical)
                if physical['safe']:row.update(scan_arena(a,ready,data))
                if rows and 'ordinary' in row:row.update(compare(rows[0],row))
                rows.append(row)
                print(angle,physical['safe'],physical['base_contact_m'],row.get('ordinary_covered'),
                      [(r['id'],None if r['best_m'] is None else round(r['best_m']*1000)) for r in row.get('low',())],
                      row.get('live_eligible'),flush=True)
                _write(out,dict(plan_sha256=_hash(procedure),low_sha256=_hash(PREPARED),rows=rows))
            finally:a.close()
    options=[r for r in rows if r.get('live_eligible')]
    pick=max(options,key=lambda r:(sum(r['low_improvements_m']),r['ordinary_covered']))['angle'] if options else None
    _write(out,dict(plan_sha256=_hash(procedure),low_sha256=_hash(PREPARED),rows=rows,chosen=pick))


def unchanged(m,angle):
    geometries=np.ones(m.ngeom,dtype=bool)
    if angle:geometries[m.geom(ADAPTER).id]=False
    fixed=m.body('tt20_table_end_clamp').id
    bodies=np.arange(m.nbody)!=fixed
    return ({f:np.copy(getattr(m,f)) for f in (
                'jnt_axis','jnt_range','dof_damping','actuator_ctrlrange','actuator_forcerange','eq_data',
                'cam_pos','cam_quat')}
            |{f:np.copy(getattr(m,f)[bodies]) for f in ('body_inertia','body_mass','body_ipos','body_iquat')}
            |{f:np.copy(getattr(m,f)[geometries]) for f in (
                'geom_pos','geom_quat','geom_size','geom_type','geom_friction','geom_contype','geom_conaffinity')})


def live(plan_file,grip_file,geometry_file,out,mode,condition):
    d=_read(plan_file);grip=_read(grip_file);c=_read(geometry_file)
    if (d['parent_sha256']!=_hash(grip_file) or d['cards']!=grip['cards']
            or d['ready']!=grip['ready'] or d['cap']!=grip['cap']
            or c['plan_sha256']!=_hash(plan_file)
            or grip['source']['bank_sha256']!=_hash(BANK) or grip['source']['optics_sha256']!=_hash(OPTS)):
        raise ValueError('Auxiliary clamp changed its standalone protocol or fixed memory')
    angle=0. if condition=='stock' else c['chosen']
    if angle is None:raise ValueError('No adapter passed the independent mechanical qualification')
    physical_live(d,out,mode,condition,pitched(angle),audit_wrist=False,
                  metadata=dict(adapter=parameters(angle),plan_sha256=_hash(plan_file),census_sha256=_hash(geometry_file)))


def summarize(procedure,out,files):
    p=_read(procedure);series={}
    for source in files:
        d=_read(source);name=f"{d['mode']}-"+('stock' if d['adapter']['degrees']==0. else 'alternate')
        if d['plan_sha256']!=_hash(procedure) or len(d['rows'])!=len(d['cards']) or d['cards']!=p['cards'] or name in series:
            raise ValueError('The assembled physical comparison is incomplete or unmatched')
        points=[]
        for card,r in zip(d['cards'],d['rows'],strict=True):
            score=r['score'];delivered=r['placements']['delivered']
            moderate=[q for q in delivered if q.get('receive_line') and .99<=projected_height(q['receive_line'])<=1.04]
            first=delivered[0] if delivered else None
            points.append(dict(id=card['id'],own=score['learner'],far=score['opponent'],total=score['returns'],
                    limit=score['returns']>=p['cap'],first_placement=first['target'] if first else None,
                    first_bounce=first['near_bounce']['location'] if first else None,
                    moderate_incoming=len(moderate),moderate_returned=sum(q['near_landed'] for q in moderate),
                    terminal_projected_m=projected_height(delivered[-1]['receive_line']) if delivered and
                    delivered[-1]['receive_line'] and not delivered[-1]['near_landed'] else None,
                    cpu=r['runtime_cpu_s'],clear=r['clearance'],memory=r['memory']))
        totals={k:sum(r[k] for r in points) for k in ('own','far','total','moderate_incoming','moderate_returned','cpu','limit')}
        totals.update(clearance_states=sum(r['clear']['samples'] for r in points),
                      penetration_states=sum(r['clear']['penetration_samples'] for r in points))
        series[name]=dict(receipt_sha256=_hash(source),points=points,totals=totals)
    if set(series)!={'raw-stock','raw-alternate','frozen-stock','frozen-alternate'}:
        raise ValueError('Both body configurations and both evaluation policies are required')
    matched={}
    for mode in ('raw','frozen'):
        a,b=(series[f'{mode}-{key}']['points'] for key in ('stock','alternate'))
        matched[mode]=['no_legal_offer_either' if x['first_bounce'] is None and y['first_bounce'] is None else
                       'same_legal_offer' if x['first_placement']==y['first_placement']
                       and x['first_bounce'] is not None and y['first_bounce'] is not None
                       and float(np.linalg.norm(np.asarray(x['first_bounce'])-y['first_bounce']))<.00001
                       else 'different' for x,y in zip(a,b,strict=True)]
    _write(out,dict(plan_sha256=_hash(procedure),series=series,opening_match=matched))


def audit(out,files):
    rows=[]
    for source in files:
        d=_read(source)
        if len(d['rows'])!=len(d['cards']):raise ValueError('Incomplete course')
        i=max(range(len(d['rows'])),key=lambda k:d['rows'][k]['score']['learner'])
        card,row=d['cards'][i],d['rows'][i];angle=d['adapter']['degrees']
        with centered(d['ready']),pitched(angle):
            a=arena();cad=arena(detail='cad');touches=[]
            try:
                a.launch(row['shot'],card=card);a.partner.reset(seed=card['seed'])
                guarded(a,CourtClearance(cad.model,articulated=True))
                with contact_observer(a,touches):
                    for u in row['actions']:a.exchange(u,FAR_COMMAND)
                verify_clearance(a)
                actual=(a.events,a.point_placements(),score_events(a.events,opening_side=1),a.clearance_result())
                expected=tuple(row[k] for k in ('events','placements','score','clearance'))
                if tuple(digest_json(o) for o in actual)!=tuple(digest_json(o) for o in expected):
                    raise ValueError('Physical repeat changed')
                if len(touches)!=sum(e['kind']=='contact' for e in a.events):
                    raise ValueError('Native contact audit missed a referee hit')
                rows.append(dict(id=card['id'],mode=d['mode'],source_sha256=_hash(source),angle=angle,
                    score=row['score'],exact_events=True,contacts=touches,clearance=a.clearance_result()))
            finally:a.close();cad.close()
    _write(out,dict(rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','geometry','raw','frozen','summarize','audit'))
    p.add_argument('out',type=Path)
    p.add_argument('--grip-plan',type=Path,default=ROOT/'tt41/shallow-grip-plan.json')
    p.add_argument('--plan',type=Path,default=ROOT/'tt41/shallow-clamp-plan.json')
    p.add_argument('--geometry',type=Path,default=ROOT/'tt41/shallow-clamp-geometry.json')
    p.add_argument('--condition',choices=('stock','alternate'),default='alternate')
    p.add_argument('--receipts',type=Path,nargs=4)
    p.add_argument('--audit-receipts',type=Path,nargs='+')
    a=p.parse_args()
    if a.mode=='plan':plan(a.grip_plan,a.out)
    elif a.mode=='geometry':geometry(a.plan,a.out)
    elif a.mode=='summarize':summarize(a.plan,a.out,a.receipts)
    elif a.mode=='audit':audit(a.out,a.audit_receipts)
    else:live(a.plan,a.grip_plan,a.geometry,a.out,a.mode,a.condition)
