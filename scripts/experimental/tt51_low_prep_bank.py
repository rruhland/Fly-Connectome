"""Export earlier RGB-cued, physically rewarded preparation experience for reuse."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .tt51_low_body_bridge import BodyContinuation
from .tt51_low_reverse_curriculum import arm,plain

OUT=Path('artifacts/tt51_low_reverse/observed-prep-bank.json.gz')


class EncoderPreparation(BodyContinuation):
    """The executed bridge metric on a wider observed bank; no source clocks."""
    def __init__(self,packet):
        self.rows=[]
        for row in packet['trajectories']:
            self.rows.append(dict(name=row['name'],q=np.asarray(row['q']),
                ref=np.asarray(row['ref']),next=np.asarray(row['next_ref'])))
        self.reset()

    def preview(self,q,ref):
        """Read-only supported entry and remaining demonstrated motor path."""
        q=np.asarray(q);ref=np.asarray(ref);out=[]
        for i,row in enumerate(self.rows):
            d=np.mean(((row['q'][:,:5]-q[:5])/.12)**2,axis=1)
            d+=np.mean(((row['ref'][:,:5]-ref[:5])/.16)**2,axis=1)
            at=int(np.argmin(d))
            out.append(dict(trajectory=i,phase=at,body_distance=float(d[at]),supported=bool(d[at]<=3.),
                remaining_ticks=len(row['q'])-1-at,remaining_refs=row['next'][at:].copy(),
                endpoint_q=row['q'][-1].copy(),endpoint_ref=row['next'][-1].copy(),
                minimum_reference_travel_seconds=float(np.max(abs(row['next'][-1,:5]-ref[:5]))/4.16)))
        return sorted(out,key=lambda r:r['body_distance'])


def export(output=OUT):
    packet=arm._read(arm.SKILL);late=arm._read(Path('artifacts/tt51_low_reverse/course.json.gz'))
    doc=dict(schema='tt51_observed_rewarded_preparation_bank_v1',trajectories=[],
        source_sha256=arm._hash(arm.SKILL),late_templates_sha256=arm.digest_json(late['initial_templates']),
        source_code_sha256=arm._hash(Path(__file__)),late_templates=late['initial_templates'],
        input_scope='Only actual RGB, own encoders and own issued references. Physical scalar legal reward admits source. Separate source tick provenance is not read by EncoderPreparation.',
        activation_scope='Bank does not classify future incoming low or authorize blind full preparation; caller needs causal scene/body/time support. Wider bank entries have not yet been physically retried.',
        replay_evidence='Earlier4034/4050/4060 entries tested with actual physics in stock-bridge.json.gz; widening to observed4010 adds recorded experience, no new executed motor result.')
    for i,sample in enumerate(packet['rows']):
        if sample['reward']!='physical legal and answered':raise ValueError('Source lacks physical reward')
        start=next(r['tick'] for r in sample['observations'] if r['x'] is not None
            and r['x'][4]>.9 and r['x'][1]>-.99)
        cut=(4079,4062)[i]
        rows=[r for r in sample['observations'] if start<=r['tick']<=cut]
        doc['trajectories'].append(dict(name=f'earned-preparation-{i}',
            q=[r['q'] for r in rows],ref=[r['ref'] for r in rows],next_ref=[r['next_ref'] for r in rows],
            rgb=[r['x'] for r in rows],late_template=deepcopy(late['initial_templates'][i]),
            provenance=dict(source=sample['name'],first_observed_outbound_tick=start,last_observed_prep_tick=cut,
                source_ticks=[r['tick'] for r in rows],admission_reward=sample['reward'])))
    arm._write(output,plain(doc));print(output,[len(r['q']) for r in doc['trajectories']],flush=True)


if __name__=='__main__':export()
