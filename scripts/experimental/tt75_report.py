"""Report assigned physical games and observation cost without selecting results."""
import argparse
from pathlib import Path
import numpy as np
from . import tt75_ordinary_play as tt
from .tt4_transfer_trial import _read,_write


def totals(rows):
 delivered=[r for g in rows for r in g['placements']['delivered']]
 legal=[d for d in delivered if d['near_bounce'] is not None]
 return dict(games=len(rows),learner_returns=sum(r['score']['learner'] for r in rows),
             opponent_legal=sum(r['score']['opponent'] for r in rows),
             actual_far_contacts=len(delivered),actual_legal_offers=len(legal),
             answered_legal_offers=sum(d['near_landed'] for d in legal),
             first_offered=sum(bool(g['placements']['delivered']) and g['placements']['delivered'][0]['near_bounce'] is not None for g in rows),
             first_answered=sum(bool(g['placements']['delivered']) and g['placements']['delivered'][0]['near_landed'] for g in rows),
             actual_control_ticks=sum(len(g['actions']) for g in rows),
             actor_loop_cpu_seconds=sum(g['runtime_cpu_s'] for g in rows))


def report(path,out):
 data=_read(path);base=_read(tt.BASELINE);rows=data['rows'];n=len(rows)
 if data.get('invalid_instrumentation'):
  raise ValueError('Instrumentation failure is not a sensing-policy assessment')
 if data['cards'][:n]!=base['cards'][:n]:raise ValueError('Assigned cards differ')
 delta=[r['score']['learner']-b['score']['learner'] for r,b in zip(rows,base['rows'])]
 result=dict(complete_fixed32=n==32,source=str(path),baseline=str(tt.BASELINE),
             observed=totals(rows),parent=totals(base['rows'][:n]),
             paired_game_deltas=delta,paired_gains=sum(d>0 for d in delta),paired_losses=sum(d<0 for d in delta),
             total_wall_seconds=data.get('wall_seconds'),total_cpu_seconds=data.get('cpu_seconds'))
 first_changes=[]
 for i,(row,old) in enumerate(zip(rows,base['rows'])):
  current,previous=row['placements']['delivered'],old['placements']['delivered']
  if not current or not previous or current[0]['near_landed']==previous[0]['near_landed']:continue
  a,b=np.asarray(row['actions']),np.asarray(old['actions']);length=min(len(a),len(b))
  changed=np.flatnonzero(np.any(a[:length]!=b[:length],axis=1))
  if not len(changed):continue
  tick=int(changed[0]);when=tick/60
  before=[[e for e in tape['events'] if e['time']<=when+1.e-8] for tape in (row,old)]
  shared=tt.held.school.digest_json(before[0])==tt.held.school.digest_json(before[1])
  far_contact=previous[0]['far_contact']['time']
  first_changes.append(dict(index=i,first_different_command=tick,
      actual_first_offer_rescued=bool(current[0]['near_landed']),
      shared_exact_physical_event_prefix=shared,
      first_far_contact_before_control_divergence=far_contact<=when,
      same_incoming_causal_comparison=bool(shared and far_contact<=when)))
 result['first_offer_changes']=first_changes
 observed=[r['observer'] for r in rows if 'decisions' in r.get('observer',{})]
 if observed:
  decisions=[d for o in observed for d in o['decisions']]
  eligible=[d for d in decisions if d['x'] is not None and d['x'][4]<-.9 and -1.62<d['x'][1]<-.49 and d['x'][2]>.77]
  camera=sum(o['camera_calls'] for o in observed);cpu=sum(o['camera_cpu_seconds']+o['fit_cpu_seconds'] for o in observed)
  result['observer']=dict(decisions=len(decisions),dense=sum(d['source']=='dense' for d in decisions),
      legacy=sum(d['source']=='legacy' for d in decisions),selected_search_window_decisions=len(eligible),
      dense_selected_search_window=sum(d['source']=='dense' for d in eligible),
      camera_calls=camera,valid_stereo=sum(o['valid_stereo'] for o in observed),
      camera_cpu_seconds=sum(o['camera_cpu_seconds'] for o in observed),fit_cpu_seconds=sum(o['fit_cpu_seconds'] for o in observed),
      sensor_cpu_seconds_per_actual_legal_offer=None if not result['observed']['actual_legal_offers'] else cpu/result['observed']['actual_legal_offers'])
 if out:_write(out,result)
 return result


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--out',type=Path)
 a=p.parse_args();print(report(a.receipt,a.out))
