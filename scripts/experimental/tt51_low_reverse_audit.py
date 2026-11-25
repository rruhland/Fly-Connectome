"""Independent issued-command replay with full-CAD checking, shared prefix paid once."""
import argparse
from pathlib import Path
from time import monotonic,process_time
from unittest.mock import patch
from .tt51_low_reverse_curriculum import arm,inputs
from . import tt37_opponent as opponent
from .tt37_opponent_audit import CourtClearance,guarded,verify_clearance


def run(source,output,stages,indices):
    data=arm._read(source);src=arm._read(arm.ARCHIVE);case,original=arm.terminal(src)
    _,teachers=inputs();root=arm._read(arm.PLAN)['anchor']
    document=dict(schema='tt51_low_reverse_full_cad_replay_v1',source_sha256=arm._hash(source),
        controller_source_sha256=arm._hash(Path(__file__)),rows=[],complete=False,
        physical_replay_only=True,stages=stages,indices=indices,paid_shared_prefix_ticks=root)
    clock=monotonic();cpu=process_time();paid=0
    with arm.mpc.centered(src['ready']):
        a=arm.mpc.arena();cad=arm.mpc.arena(detail='cad');clear=CourtClearance(cad.model,articulated=True)
        try:
            a.launch(original['shot'],card=case['card']);a.partner.reset(seed=case['card']['seed']);guarded(a,clear)
            for command in original['actions'][:root]:a.exchange(command,arm.mpc.FAR_COMMAND);paid+=1
            state=arm.State(a)
            for row in data['rows']:
                if stages and row['stage'] not in stages:continue
                if indices and row['index'] not in indices:continue
                teacher=teachers[row['teacher']];request=teacher['tape']['placements']['requested'][case['leg']]
                def draw(card,leg,protocol,rng):
                    if leg==case['leg'] and row.get('random_reply_stream') is None:
                        label=('tt51_low_center_practice' if data['schema']=='tt51_low_stock_bridge_v1'
                               else 'tt51_reverse_goal_practice')
                        return tuple(row['goal']),request['draw'] if not any(row['delta']) else label
                    return arm.draw(card,leg,protocol,rng)
                for mode,result in row['variants'].items():
                    state.restore(a);a.card=dict(case['card'],reply_seed=row.get('random_reply_stream') or
                        teacher.get('stream',case['card']['reply_seed']))
                    commands=teacher['tape']['actions'][root:row['cut']]+result['controls']
                    finished=False
                    with patch.object(opponent,'draw',draw):
                        for command in commands:
                            if finished:raise RuntimeError('Recorded commands step after point completion')
                            _,_,done,capped,_=a.exchange(command,arm.mpc.FAR_COMMAND);paid+=1
                            finished=done or capped
                    verify_clearance(a)
                    exact=arm.digest_json(a.events)==result['event_sha256']
                    if not exact:raise RuntimeError('Physical event replay differs')
                    record=dict(index=row['index'],stage=row['stage'],mode=mode,exact_physical_events=exact,
                        legal=result['legal'],answered=result['answered'],clearance=a.clearance_result(),
                        paid_branch_control_ticks=len(commands))
                    document['rows'].append(record)
                    document.update(actual_control_ticks=paid,wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu)
                    arm._write(output,document)
                    print(row['index'],mode,result['legal'],record['clearance']['penetration_samples'],flush=True)
        finally:a.close();cad.close()
    document['complete']=True;arm._write(output,document)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--stage',action='append',default=[]);p.add_argument('--index',action='append',type=int,default=[])
    a=p.parse_args();run(a.source,a.output,a.stage,a.index)
