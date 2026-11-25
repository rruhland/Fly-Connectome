"""Use same-offer lateral-only controls to isolate the whole installation's contribution."""

from pathlib import Path

from . import tt63_lateral_clamp as lateral
from . import tt84_installation_system as full
from .tt4_transfer_trial import _hash, _read, _write


OUT=full.ROOT/'lateral-control.json.gz'


def run():
    p=full.contract(); plan=_read(lateral.HOLD); g=_read(lateral.GEOMETRY)
    ready=next(r['ready'] for r in g['rows'] if r['dx']==.12)
    src={r['case']['delivery']:r for r in plan['rows'] if r['dx']==.12}
    receipt=dict(protocol=_hash(full.PLAN),lateral_reference=_hash(lateral.HOLD),lateral_geometry=_hash(lateral.GEOMETRY),rows=[],complete=False)
    if OUT.exists():
        old=_read(OUT)
        if any(old[k]!=v for k,v in receipt.items() if k not in ('rows','complete')):
            raise ValueError('Changed lateral benchmark')
        receipt=old
    with full.fixture.school.enabled(cameras=False),lateral.slid(.12),full.MP.centered(ready):
        live=full.MP.arena(width=64,height=48); cad=full.MP.arena(width=64,height=48,detail='cad')
        try:
            guard=full.q.GripClearance(cad.model,stock=True)
            for case in p['physical'][len(receipt['rows']):]:
                key=case['delivery']; known=src.get(key)
                donor=None
                if known is not None:
                    a=known['case']
                    if any(a[k]!=case[k] for k in ('source','id','i','role')):
                        raise ValueError('Side translation controls have physically different score contexts')
                    trial=known['motion'];donor=key
                else:
                    doc=_read(Path(case['source']))
                    trial=full.q.score(live,guard,doc,case,'lateral',3)
                row=dict(case=case,trial=trial,reused=donor,real=any(full.q.received(t) for t in trial['physical']),
                         broad=any(full.q.received(t,True) for t in trial['physical']),reference=full.q.received(trial['stock_commands']))
                receipt['rows'].append(row);receipt['complete']=len(receipt['rows'])==len(p['physical']);_write(OUT,receipt)
                print(case['id'],case['role'],row['real'],row['broad'],row['reference'],donor is not None,flush=True)
        finally:live.close();cad.close()


if __name__=='__main__':run()
