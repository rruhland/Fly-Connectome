"""Native frozen policy audit with exact existing public visual-ID metadata.

No private simulator identity or new geometry enters policy inputs. Public entity
snapshots retain every required field, including None versus zero velocity age.
"""
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from stateful_pong_comparison import StatefulEvaluator
from pong_fixed_rbf import load_readout
from pong_relational_persistent_binding import public_rows,PUBLIC_FIELDS
import pong_relational_persistent_adapter as adapter
from run_rally_option_memory import MODEL,EXPECTED,read,write,sha,seed_rngs
from return_metrics import ReturnCounter,native_count_status
from rally_raw_capture import SparseRows
from exact_visual_transport_copy import install_exact_transport_copy


def snapshot(entities):
    rows,metadata=public_rows(entities)
    return [dict(**meta,normalized_position=row[:2].tolist(),observed_velocity=row[2:4].tolist())
        for row,meta in zip(rows,metadata)]


def save_public_entities(path,observations):
    rows=[];metadata=[];lengths=[]
    for observation in observations:
        row,meta=public_rows(observation);rows.extend(row);metadata.extend(meta);lengths.append(len(row))
    ids=np.asarray([m['id'] for m in metadata],np.int64)
    if np.any(ids<0):raise ValueError('public ID overflow in int64 archive')
    np.savez_compressed(path,entities=np.asarray(rows,np.float64).reshape(-1,6),indptr=np.r_[0,np.cumsum(lengths)],ids=ids,
        observation_age_seconds=np.asarray([m['observation_age_seconds'] for m in metadata]),
        velocity_age_seconds=np.asarray([0. if m['velocity_age_seconds'] is None else m['velocity_age_seconds'] for m in metadata]),
        velocity_age_valid=np.asarray([m['velocity_age_seconds'] is not None for m in metadata],bool),
        association_strength=np.asarray([m['association_strength'] for m in metadata]))


def load_public_entities(path):
    with np.load(path,allow_pickle=False) as z:
        rows=z['entities'];observations=[]
        for a,b in zip(z['indptr'][:-1],z['indptr'][1:]):
            entities=[dict(id=int(z['ids'][i]),normalized_position=rows[i,:2].tolist(),
                observed_velocity=rows[i,2:4].tolist(),observed=bool(rows[i,4]),velocity_known=bool(rows[i,5]),
                observation_age_seconds=float(z['observation_age_seconds'][i]),
                velocity_age_seconds=float(z['velocity_age_seconds'][i]) if z['velocity_age_valid'][i] else None,
                association_strength=float(z['association_strength'][i])) for i in range(int(a),int(b))]
            public_rows(entities);observations.append(entities)
    return observations


class PublicEntityEvaluator(StatefulEvaluator):
    def __init__(self,*args,**kwargs):
        self.entity_inputs=[];super().__init__(*args,**kwargs)
    def _choose_policy(self):
        baseline,q=self.rbf_readout.choose(*policy_features(self.latest_encoded,self.scaler))
        entities=self.latest_encoded['entities']
        self.entity_inputs.append(snapshot(entities))
        action,choice=self.adapter.choose(self.controller,entities,baseline,q)
        self.decisions.append(choice);return action,q


def audit_public_trace(model,baseline,records,decisions,inputs,raw):
    if not len(records)==len(decisions)==len(inputs)==raw.shape[0]:raise ValueError('all actual public-input rows required')
    controller=adapter.new_controller(model);changed=0;modes={};carry=0
    for i,(entities,row,expected) in enumerate(zip(inputs,records,decisions)):
        lo,hi=raw.indptr[i:i+2];base,q=baseline.choose(raw.indices[lo:hi],raw.data[lo:hi])
        action,decision=adapter.choose(controller,entities,base,q)
        if action!=row['action'] or decision!=expected or not np.array_equal(q,np.asarray(row['q'])):
            raise AssertionError(('public-entity stateful replay mismatch',i))
        adapter.observe(controller,row);changed+=action!=base
        modes[decision['mode']]=modes.get(decision['mode'],0)+1
        carry+=decision['binding_mode']=='persistent_visual_carry'
    return dict(passed=True,rows=len(records),changed_actions=int(changed),mode_counts=modes,
        persistent_binding_cameras=int(carry),macro_starts=0,held_cameras=0,new_cameras=0,
        public_metadata_fields=list(PUBLIC_FIELDS),all_policy_decisions_replayed=True)


def native_task(model_path,directory,seed,index,deadline):
    began=time.monotonic();directory=Path(directory);seed_rngs()
    if time.monotonic()>=deadline:return dict(seed=seed,complete=False,not_started=True,transitions=0,initial_cameras=0)
    baseline=load_readout(MODEL/'rbf-model.npz');model=adapter.load_model(model_path)
    session=PublicEntityEvaluator(baseline,model,adapter,PongEvaluationConfig(seed=int(seed),episodes=1,
        max_episode_ticks=2400,opponent_speed=.65,policy_scaling=read(MODEL/'radial-cap.json')))
    transport_copy=install_exact_transport_copy(session)
    if time.monotonic()>=deadline:return dict(seed=seed,complete=False,not_started=True,transitions=0,initial_cameras=0)
    session._begin_episode();game=session.env.game;original=game.step;counter=ReturnCounter();raw=SparseRows();records=[]
    def instrument(drive,**kwargs):return counter.step(game,original,drive,**kwargs)
    game.step=instrument
    try:
        while not session.complete and time.monotonic()<deadline:
            raw.add(session);records.append(session.step(check_frozen=False))
    finally:del game.step
    session.verify_frozen();report=session.summary();last=report['rallies'][-1] if report['rallies'] else dict(
        truncated=False,terminated=False,reward=session.episode_return,seconds=session.episode_seconds)
    annotation=counter.summary(game,**{k:last[k] for k in ('truncated','terminated','reward','seconds')})
    stem=directory/f'candidate_{index:02d}';X=raw.matrix();sp.save_npz(str(stem)+'-raw.npz',X)
    public_path=Path(str(stem)+'-public-entities.npz');save_public_entities(public_path,session.entity_inputs)
    restored=load_public_entities(public_path)
    audit=audit_public_trace(model,baseline,records,session.decisions,restored,X);session.verify_frozen()
    result=dict(seed=int(seed),complete=session.complete,records=records,decisions=session.decisions,report=report,
        annotation=annotation,full_stateful_audit=audit,transitions=session.transitions,initial_cameras=session.initial_cameras,
        model_sha256=sha(model_path),baseline_sha256=EXPECTED,transport_kind=adapter.CONTRACT,
        require_first_hit=model.require_first_hit,feature_file=str(stem)+'-raw.npz',
        transport_copy=transport_copy,
        public_feature_file=str(public_path),structured_feature_file=str(public_path),
        public_metadata_schema='exact-public-visual-entities-float64-int64-validity-v1',
        public_metadata_sha256=sha(public_path),wall_seconds=time.monotonic()-began)
    if report['complete']:assert native_count_status(annotation,report)['valid_terminal_counts']
    write(str(stem)+'.json',result);return result
