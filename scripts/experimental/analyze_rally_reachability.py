"""Privileged saved-trace diagnostics only; never a learner input or action label.

Approximate extreme-drive reach bounds and nearest-belief checks. No new rollout.
"""
import argparse
import json,math
from pathlib import Path
from collections import defaultdict
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.directory
rows=[]
def endpoint(d,t,drive):
 y=d['paddle'];v=d['paddle_velocity'];a=d['paddle_activation'];dt=1/120
 for _ in range(max(0,round(t/dt))):
  a+=(drive-a)*(1-math.exp(-dt/.08));v=max(-.7,min(.7,v+dt*(-3*a-4*v)));y+=dt*v
  if y<=.1 or y>=.9:y=max(.1,min(.9,y));v=0.
 return y
for p in sorted((root/'episodes').glob('train_*.json')):
 r=json.loads(p.read_text())
 for e in r['opportunity_events']:
  if e['kind']!='own':continue
  for lead in [5,10,20,40,80]:
   i=e['transition']-lead
   if i<0:continue
   d=r['diagnostic_physics_only'][i];x,y=d['ball'];vx,vy=d['ball_velocity']
   if vx>=0:continue
   t=max(0,(x-.0625)/-vx);raw=y-.0125+vy*t;span=.975;z=raw%(2*span);target=.0125+(z if z<=span else 2*span-z)
   lo=endpoint(d,t,1);hi=endpoint(d,t,-1);reachable=(target>=lo-.1125 and target<=hi+.1125)
   entities=r['belief_trace'][i];nearest=min(entities,key=lambda q:(q['position'][0]-(64*y-.5))**2+(q['position'][1]-(64*x-.5))**2) if entities else None
   dist=math.dist(nearest['position'],[64*y-.5,64*x-.5]) if nearest else 999
   tracked=bool(nearest and dist<3)
   rows.append(dict(seed=r['seed'],ordinal=e['ordinal'],success=e['success'],lead=lead,reachable=reachable,gap=abs(target-d['paddle']),time_to_plane=t,tracked=tracked,observed=bool(tracked and nearest['observed']),known=bool(tracked and nearest['known']),position_error=dist,velocity_error=math.dist(nearest['velocity'],[vy,vx]) if tracked and nearest['known'] else None))
summary={}
for later in [False,True]:
 for success in [False,True]:
  for lead in [5,10,20,40,80]:
   v=[a for a in rows if (a['ordinal']>1)==later and a['success']==success and a['lead']==lead]
   if not v:continue
   summary[f'{"later" if later else "first"}_{"return" if success else "miss"}_lead{lead}']=dict(n=len(v),reachable=sum(a['reachable'] for a in v),tracked=sum(a['tracked'] for a in v),observed=sum(a['observed'] for a in v),known=sum(a['known'] for a in v),mean_gap=sum(a['gap'] for a in v)/len(v),mean_t=sum(a['time_to_plane'] for a in v)/len(v),mean_velocity_error=sum(a['velocity_error'] for a in v if a['velocity_error'] is not None)/max(1,sum(a['velocity_error'] is not None for a in v)))
result=dict(episodes=len(list((root/'episodes').glob('train_*.json'))),diagnostic_only=True,no_policy_observations_executed=True,method='privileged analytic reflected impact and approximate constant-extreme-drive reach bounds; nearest sensor entity within3px; not learning labels',summary=summary)
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
