"""One fixed hit-learning screen after the selected-lineage continuity repair."""
import argparse
from dataclasses import asdict
import fcntl,json,time
from pathlib import Path
import numpy as np
import run_reliable_return_learning as previous
import run_dct_sarsa_pong as base
import run_pong_policy_search_continuation as rollout
import run_pong_development as worker
from fly_connectome.control.session import source_signature,MODEL_HASH,VISION_CONFIGURATION
from fly_connectome.control.durable import runtime_signature
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.projection import FixedPolicyProjection
from fly_connectome.control.rewards import REWARD_VERSION
from return_metrics import aggregate,paired_evidence

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/reconstruction/2026-10-21-continuity-repair-protocol.md'
TRAIN_SECONDS=1200.;TOTAL_SECONDS=1800.;SERVES=32
CONTROLS=('initial','up','noop','random1','random10')
sha=base.sha


def serve_schedule():
    values=np.random.default_rng(np.random.SeedSequence([1287749219,29103])).integers(1,2147483647,size=SERVES)
    result=[int(x) for x in values]
    if len(set(result))!=SERVES or set(result)&set(base.FINAL):
        raise ValueError('fresh development stream collision')
    return result


def declared_jobs():
    learning=[dict(id='learn_corrected',kind='learn',seed=736187003,alpha=.001,lam=.995,
        opponent_speed=.65,action_hold_frames=1,exploration_block_frames=1,
        transitions=24000,checkpoints=[24000],projection_from='projection.json',compact_reports=True,
        max_abs_q=1000.,reward_shaping=dict(version=REWARD_VERSION,score_coefficient=0.,
            own_hit_bonus=1.,survival_per_second=0.))]
    evaluation=[]
    for policy in ('corrected',*CONTROLS):
        trained=policy=='corrected'
        for i,seed in enumerate(serve_schedule()):
            kind='trained' if trained else ('random' if policy.startswith('random') else policy)
            action_seed=int(np.random.SeedSequence([447201113,1287749219,i]).generate_state(1)[0]) if kind=='random' else None
            config=asdict(PongEvaluationConfig(seed,episodes=1,max_episode_ticks=1200,policy_kind=kind,
                action_seed=action_seed,skip_baseline_perception=not trained,opponent_speed=.65,
                action_hold_frames=10 if policy=='random10' else 1))
            evaluation.append(dict(id=f'eval_{policy}_s{i:02d}',kind='evaluate',policy=policy,serve_index=i,
                config=config,weights_from='learn_corrected_at24000.npy' if trained else None,
                scaling_from='learn_corrected_at24000.cap.json' if trained else None,compact_reports=True))
    return learning,evaluation


def prepare(directory):
    directory.mkdir(parents=True,exist_ok=True)
    descriptor=json.loads((directory/'projection.json').read_text())
    projection=FixedPolicyProjection(4389,descriptor);projection.validate_embedded()
    if (descriptor['file_sha256']!=sha(base.TRANSFORM) or descriptor['max_norm2']!=256.
            or descriptor['shape']!=[214,4389]
            or descriptor['calibration']['fit_sha256']!=sha(base.FIT)):
        raise ValueError('exact frozen DCT/fit and cap required')
    validation=json.loads((directory/'continuity-validation.json').read_text())
    if (not validation['passed'] or validation['source']!=source_signature()
            or validation['vision_configuration']!=VISION_CONFIGURATION):
        raise ValueError('continuity validation differs from the actual visual service')
    learning,evaluation=declared_jobs()
    expected=dict(source=source_signature(),runtime=runtime_signature(),model_hash=MODEL_HASH,
        vision_configuration=VISION_CONFIGURATION,jobs=learning+evaluation,
        driver_hash=sha(worker.__file__),coordinator_hash=sha(__file__),protocol_hash=sha(PROTOCOL),
        completion_helper_hash=sha(base.__file__),rollout_helper_hash=sha(rollout.__file__),
        supervisor_helper_hash=sha(previous.__file__),metrics_hash=sha(Path(__file__).with_name('return_metrics.py')),
        validation_hash=sha(directory/'continuity-validation.json'),projection_hash=sha(directory/'projection.json'),
        matrix_file_hash=sha(base.TRANSFORM),fit_hash=sha(base.FIT),serve_seeds=serve_schedule(),
        training_seed=736187003,train_wall_seconds=TRAIN_SECONDS,overall_wall_seconds=TOTAL_SECONDS,
        training_camera_cap=24000,evaluation_camera_cap=96000,training_workers=1,evaluation_workers=4,
        calibration_rows_reused=1024,new_calibration_observations=0,long_bank_entries=0,context_fallback_steps=4,
        reserved_final_seeds=list(base.FINAL))
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        if any(manifest[k]!=v for k,v in expected.items()):raise ValueError('registered repaired-vision study differs')
    else:
        manifest=dict(**expected,started_wall_ns=time.time_ns());worker.atomic_json(path,manifest)
    return manifest,learning,evaluation


def supervise(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,learning,evaluation=prepare(directory);start=manifest['started_wall_ns']/1e9
        training=[previous.execute_training(directory,learning[0],start+TRAIN_SECONDS)]
        ready=training[0]['status']=='complete'
        eligible=[j for j in evaluation if j['policy']!='corrected' or ready]
        evaluated=previous.execute_evaluation(directory,eligible,start+TOTAL_SECONDS-120.)
        annotations=previous.annotate(directory,[*learning,*eligible])
        worker.atomic_json(directory/'return-annotations.json',annotations)
        groups={};by_policy={}
        for policy in ('corrected',*CONTROLS):
            rallies=[r for j in eligible if j['policy']==policy and j['id'] in annotations
                     for r in annotations[j['id']]['rallies']]
            by_policy[policy]=rallies;groups[policy]=dict(complete=len(rallies)==SERVES,**aggregate(rallies))
        counts=dict(learn=sum(base.sample_count(directory,j['id']) for j in learning),
                    evaluate=sum(base.sample_count(directory,j['id']) for j in evaluation))
        assert counts['learn']<=24000 and counts['evaluate']<=96000
        complete=ready and evaluated['complete']
        summary=dict(status='complete' if complete else 'partial',counts=counts,training=training,evaluation=evaluated,
            groups=groups,paired_evidence=paired_evidence(by_policy,arms=('corrected',),
                expected_serves=SERVES,controls=CONTROLS),wall_seconds=time.time()-start,
            annotations_file='return-annotations.json',final_holdout_unused=list(base.FINAL),
            scope='One-seed repaired-vision hit-only screen; historical comparison is nonpaired, not a standard Pong claim.')
        worker.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.directory)
    else:supervise(args.directory)
