"""One reviewed fixed-partition value pilot; one clock, no retries or sweep."""
import argparse
from pathlib import Path
import resource
import time
import numpy as np
import scipy.sparse as sp
from pong_fixed_rbf import load_readout
from pong_fixed_value_tree import (visual_features, build_forest, leaves_batch, fit_values,
    supported_values, FixedValueTreeReadout, save_readout)
from run_rally_value_collection import sha, read, write, ROOT, MODEL, BANK, EXPECTED
from run_rally_fixed_bank_check import run as native_check
from run_rally_raw_fresh_screen import run as fresh_run
from run_rally_repertoire_collection import exclusions, select_starts
import run_practical_pong_v2 as base

DEST = ROOT/'pong-fixed-value-tree-20261028-r1'
INDEX = ROOT/'pong-value-collection-20261028-r1/eligible-training-transitions.json'
BASIS = ROOT/'pong-rally-smooth-actor-20261028-r1/correction-basis.npy'
SCREEN8 = ROOT/'pong-value-feasibility-20261028-r1/screen8-bank.json'
FILES = ('run_rally_fixed_value_tree.py', 'pong_fixed_value_tree.py', 'pong_fixed_rbf.py',
    'pong_fixed_pairwise.py', 'run_rally_fixed_bank_check.py', 'run_rally_guarded_curriculum.py',
    'rally_action_equivalence.py', 'run_rally_raw_optimizer.py', 'run_rally_raw_fresh_screen.py',
    'run_rally_repertoire_collection.py', 'run_rally_value_collection.py', 'rally_raw_capture.py',
    'return_metrics.py', 'run_practical_pong_v2.py')


def metadata(item):
    n = item['eligible_transitions']
    with np.load(item['transition_file'], allow_pickle=False) as f:
        a = f['actions'].copy(); r = f['hits'].copy(); p = f['p'].copy()
        if item['source'] == 'new_epsilon_collection':
            terminal = f['terminated'].copy(); dt = f['elapsed_seconds'].copy()
            ticks = f['physics_ticks'].copy()
            assert len(a) == n and len(terminal) == n
            np.testing.assert_allclose(dt, ticks/120., rtol=0, atol=1e-12)
        else:
            receipt = read(item['receipt']); report = receipt['report']; count = len(a)
            ticks = np.diff(np.arange(count+1, dtype=np.int64)*12//5)
            terminal = np.zeros(count, dtype=bool)
            if report['complete']:
                rally = report['rallies'][0]; total = round(rally['seconds']*120)
                assert abs(rally['seconds']-total/120) < 1e-9
                last = total-int(ticks[:-1].sum()); assert 1 <= last <= ticks[-1]
                ticks[-1] = last; terminal[-1] = rally['terminated']
            assert abs(report['physics_seconds']-ticks.sum()/120) < 1e-9
            assert n == count if terminal[-1] else n == count-1
            dt = ticks/120.
        assert np.isin(a, [0,1,2]).all() and np.isin(r, [0,1]).all()
        assert p.shape == (len(a),3) and np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-13)
        assert int(r[:n].sum()) == item['reward_rows']
        assert int(terminal[:n].sum()) == item['natural_terminals']
        return a[:n], r[:n], p[:n], terminal[:n], dt[:n]


def allocate(population, protected, quota):
    result = np.asarray(protected, dtype=np.int64).copy()
    assert result.sum() <= quota <= sum(population)
    # Equal execution allocations with deterministic handling of protected/small episodes.
    while result.sum() < quota:
        available = np.flatnonzero(result < population)
        smallest = result[available].min()
        for j in available[result[available] == smallest]:
            if result.sum() == quota: break
            result[j] += 1
    return result


def prepare():
    assert not DEST.exists(); DEST.mkdir()
    assert sha(MODEL/'rbf-model.npz') == EXPECTED
    items = read(INDEX)['episodes']; groups = [[i for i,e in enumerate(items)
        if (e['source']=='new_epsilon_collection') == new] for new in (False, True)]
    assert list(map(len, groups)) == [256, 32]
    geometry_rng = np.random.default_rng(2026102882); geometry = []
    for ids in groups:
        chosen = geometry_rng.choice(ids, 2048, replace=True)
        geometry.extend([dict(episode=int(i), row=int(geometry_rng.integers(items[i]['eligible_transitions']))) for i in chosen])
    # Draw geometry before consulting actions/rewards. The same RNG then fixes topology.
    topology_rng_state = geometry_rng.bit_generator.state
    protected = []
    for e in items:
        _, r, _, terminal, _ = metadata(e)
        protected.append(np.flatnonzero((r != 0) | terminal))
    totals = [sum(len(protected[i]) for i in ids) for ids in groups]
    assert sum(totals) <= 12000
    quotas = [6000, 6000]
    for g in (0,1):
        if totals[g] > quotas[g]: quotas[g] = totals[g]; quotas[1-g] = 12000-quotas[g]
    rng = np.random.default_rng(2026102885); sampling = []
    for g,ids in enumerate(groups):
        allocation = allocate([items[i]['eligible_transitions'] for i in ids], [len(protected[i]) for i in ids], quotas[g])
        for i,m in zip(ids, allocation):
            n = items[i]['eligible_transitions']; keep = protected[i]
            remaining = np.setdiff1d(np.arange(n), keep, assume_unique=True)
            draws = int(m)-len(keep)
            sampled = np.sort(np.r_[keep, rng.choice(remaining, draws, replace=False)])
            assert draws > 0 or not len(remaining), 'positive inclusion probability for all valid rows required'
            sampling.append(dict(episode=i, group=g, executions=len(ids), rows=sampled.tolist(),
                protected=keep.tolist(), unprotected_inclusion=draws/len(remaining) if len(remaining) else 1.,
                valid=n, allocation=int(m)))
    bank = read(BANK); first = [e['seed'] for e in read(SCREEN8)['bank']]
    extra = [e['seed'] for e in bank['bank'] if e['seed'] not in first]
    # Fixed 32: same LSPI eight, then first 24 existing bank starts; no outcome selection.
    screen = first+extra[:24]
    fresh = DEST/'fresh'; fresh.mkdir()
    drawn = select_starts(1943906127, 63121, exclusions(DEST), 32)
    seeds = drawn.pop('TRAIN'); drawn.update(probe=seeds[:32], unused_reserved=seeds[32:], purpose='unused paired qualification, never fit/admit')
    write(fresh/'schedules.json', drawn)
    pinned = {str(p):sha(p) for p in (INDEX, BASIS, BANK, SCREEN8, MODEL/'rbf-model.npz', MODEL/'proposal.npy', MODEL/'radial-cap.json')}
    for e in items:
        for key in ('raw_file','transition_file','receipt'):
            if key in e: pinned[e[key]] = sha(e[key])
    spec = dict(method='32fixed_state_partitions_256_supported_weighted_Bellman_sweeps',
        approved_user_architecture=True, no_backprop=True, no_privileged_inputs=True, no_expert_labels=True,
        seconds=600, fit_seconds=120, workers=4, implementation_seconds_limit=900,
        source=base.source_signature(), files={n:sha(Path(__file__).with_name(n)) for n in FILES}, pinned_assets=pinned,
        episodes=items, geometry=geometry, topology_rng_state=topology_rng_state, sampling=sampling,
        source_quotas=quotas, source_population_weights=[.5,.5], protected_rows_by_source=totals,
        tree_count=32, geometry_rows=4096, depth_limit=10, min_cache_split_rows=32,
        min_cache_child_rows=16, candidate_features_per_node=32, feature_dimension=711,
        threshold='one uniform threshold on first positive-range coordinate; failed split stops node',
        support='cell count>=4 AND ESS>=4; action >=24 of32 eligible trees; mean eligible values',
        backup='hits + .95**actual_elapsed_seconds * max supported next value; natural terminal zero; uncovered continuation zero LOWER BOUND',
        fallback='frozen40069 if no supported legal action; no blending; lowest action index tie',
        selection='final256 completed sweeps only; no sweep/restart/extension',
        screen8=first, screen32=screen, bank=str(BANK), fresh_tag=63121,
        stage8='stop if any total/later/first/second regression; continue only with an actual return gain',
        fresh_gate='strict total/later gain, retain first/second counts and later fraction',
        maximum_native_games=96, maximum_native_transition_cameras=96000,
        no_automatic_promotion=True, no_fresh_fitting=True)
    write(DEST/'spec.json',spec)
    print(dict(status='prepared_no_fit_no_native',spec_sha256=sha(DEST/'spec.json'), selected_rows=sum(s['allocation'] for s in sampling), protected=totals),flush=True)


def load_sample(spec, original, B):
    states=[]; successors=[]; actions=[]; rewards=[]; discounts=[]; weights=[]; probabilities=[]; manifest=[]
    for s in spec['sampling']:
        e=spec['episodes'][s['episode']]; X=sp.load_npz(e['raw_file']).tocsr()
        a,r,p,terminal,dt=metadata(e); idx=np.asarray(s['rows']); n=s['valid']
        assert X.shape==(e['raw_rows'],4389) and np.isfinite(X.data).all()
        nxt=np.minimum(idx+1, X.shape[0]-1)
        assert np.all((idx+1<X.shape[0]) | terminal[idx])
        pi=np.full(len(idx),s['unprotected_inclusion']);pi[np.isin(idx,s['protected'])]=1.
        w=.5/s['executions']/n/pi
        disc=.95**dt[idx];disc[terminal[idx]]=0.
        states.append(X[idx]);successors.append(X[nxt]);actions.append(a[idx]);rewards.append(r[idx]);probabilities.append(p[idx]);discounts.append(disc);weights.append(w)
        manifest.append(dict(s,reward_rows=int(r[idx].sum()), terminal_rows=int(terminal[idx].sum()),
            sum_weights=float(w.sum()), raw_sha256=sha(e['raw_file']), transitions_sha256=sha(e['transition_file'])))
    X=sp.vstack(states,format='csr');Y=sp.vstack(successors,format='csr')
    return X,Y,np.concatenate(actions),np.concatenate(rewards),np.concatenate(discounts),np.concatenate(weights),np.vstack(probabilities),manifest


def audit_native(directory, model):
    directory=Path(directory);rows=learned=changes=0;max_error=0.
    for path in sorted((directory/'native_candidate/episodes').glob('*.json')):
        receipt=read(path);feature=Path(receipt['feature_file'])
        if not feature.is_absolute():feature=directory/feature
        X=sp.load_npz(feature).tocsr();assert X.shape==(len(receipt['records']),4389)
        for i,record in enumerate(receipt['records']):
            lo,hi=X.indptr[i:i+2];ix=X.indices[lo:hi];v=X.data[lo:hi]
            action,q=model.choose(ix,v);_,support,_,_=model.inspect(ix,v)
            assert action==record['action'] and np.array_equal(q,np.asarray(record['q']))
            rows+=1;learned+=int(support.any());changes+=int(action!=model.original.choose(ix,v)[0])
    result=dict(passed=True,rows=rows,learned_decisions=learned,fallback_decisions=rows-learned,
        learned_fraction=learned/rows if rows else None,greedy_changes_from40069=changes,
        action_and_Q_rows_exact=rows,new_verification_cameras=0)
    write(directory/'tree-runtime-audit.json',result);return result


def stats(summary):
    return (summary['candidate']['own_returns'], summary['candidate_more']['later_returns'],
            summary['candidate']['first_incoming_returns'], summary['candidate_more']['tail']['2'])


def fit_and_run(approval):
    assert approval==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    spec=read(DEST/'spec.json'); began=time.monotonic(); deadline=began+600
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline,seconds=600))
    assert spec['source']==base.source_signature()
    assert all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items())
    assert all(sha(p)==h for p,h in spec['pinned_assets'].items())
    original=load_readout(MODEL/'rbf-model.npz');B=np.load(BASIS,allow_pickle=False)
    geometry=[]
    for i in sorted(set(g['episode'] for g in spec['geometry'])):
        entries=[(j,g['row']) for j,g in enumerate(spec['geometry']) if g['episode']==i]
        X=sp.load_npz(spec['episodes'][i]['raw_file']).tocsr()
        F=visual_features(X[[r for _,r in entries]],original,B)
        geometry.extend((j,f) for (j,_),f in zip(entries,F))
    G=np.asarray([f for _,f in sorted(geometry)]);rng=np.random.default_rng();rng.bit_generator.state=spec['topology_rng_state']
    table,leaf_count=build_forest(G,rng);del G,geometry
    X,Y,a,r,d,w,p,manifest=load_sample(spec,original,B)
    assert len(a)==12000 and int(r.sum())==543
    F=visual_features(X,original,B);N=visual_features(Y,original,B)
    current=leaves_batch(F,table);successor=leaves_batch(N,table)
    del N,Y
    write(DEST/'data-manifest.json',dict(traces=manifest,selected_rows=len(a),reward_rows=int(r.sum()),natural_terminals=int(np.count_nonzero(d==0)),
        cross_episode_edges=0,fresh_rows=0,weights='inverse inclusion; .5/source, equal execution and equal valid transition population weights',weight_sum=float(w.sum()),leaves=leaf_count))
    fit_began=time.monotonic();fit_deadline=began+120
    values,count,weight,ess,eligible,report=fit_values(current,successor,a,r,d,w,leaf_count,fit_deadline)
    fit_seconds=time.monotonic()-fit_began
    model=FixedValueTreeReadout(original,B,table,values,count,weight,ess,eligible)
    save_readout(DEST/'rbf-model.npz',model)
    (DEST/'proposal.npy').write_bytes((MODEL/'proposal.npy').read_bytes());(DEST/'radial-cap.json').write_bytes((MODEL/'radial-cap.json').read_bytes())
    deployed=load_readout(DEST/'rbf-model.npz');q,support,counts=supported_values(current,values,eligible)
    expected=np.argmax(np.where(support,q,-1.),axis=1);baseq=np.asarray(X@original.base.weights.T)+original.basis.batch(X)@original.residual.T
    fallback=~support.any(1);expected[fallback]=np.argmax(baseq[fallback],axis=1)
    errors=[];actual=[];inference_began=time.monotonic()
    for i in range(len(a)):
        lo,hi=X.indptr[i:i+2];ix=X.indices[lo:hi];v=X.data[lo:hi]
        action,scores=deployed.choose(ix,v);actual.append(action)
        iq,im,_,il=deployed.inspect(ix,v)
        assert np.array_equal(il,current[i]) and np.array_equal(im,support[i])
        errors.append(float(np.max(abs(iq-q[i]))))
        expected_scores=original.choose(ix,v)[1] if fallback[i] else np.where(support[i],q[i],-1.)
        np.testing.assert_allclose(scores,expected_scores,rtol=1e-12,atol=1e-12)
    assert np.array_equal(actual,expected)
    chosen_probability=p[np.arange(len(a)),actual]
    inference_start=time.monotonic()
    for i in range(min(100,len(a))):
        lo,hi=X.indptr[i:i+2];deployed.choose(X.indices[lo:hi],X.data[lo:hi])
    inference_microseconds=1e6*(time.monotonic()-inference_start)/min(100,len(a))
    report.update(status='complete_final256',fit_seconds=fit_seconds,preparation_and_parity_seconds=time.monotonic()-began,
        selected_rows=len(a),reward_rows=int(r.sum()),natural_terminal_rows=int(np.count_nonzero(d==0)),
        learned_decisions=int((~fallback).sum()),fallback_decisions=int(fallback.sum()),
        learned_fraction=float((~fallback).mean()),greedy_changes_from40069=int(np.count_nonzero(expected!=np.argmax(baseq,axis=1))),
        supported_action_counts=np.bincount(support.sum(1),minlength=4).tolist(),
        eligible_cells=int(eligible.sum()),count_quantiles=np.quantile(count,[0,.1,.5,.9,1]).tolist(),
        ESS_quantiles=np.quantile(ess,[0,.1,.5,.9,1]).tolist(),
        selected_behavior_probability_quantiles=np.quantile(chosen_probability,[0,.01,.1,.5,1]).tolist(),
        chosen_behavior_probability_below01=int(np.count_nonzero(chosen_probability<.01)),
        selected_deployed_parity_rows=len(a),maximum_value_parity_error=max(errors),
        deployed_choose_mean_microseconds_first100=inference_microseconds,
        parity_two_queries_mean_microseconds=1e6*(time.monotonic()-inference_began)/len(a),
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_sha256=sha(DEST/'rbf-model.npz'),base_weights_sha256=sha(DEST/'proposal.npy'))
    write(DEST/'fit-report.json',report);print(dict(phase='fit_parity_complete',**report),flush=True)
    assert time.monotonic()<=began+120, 'preparation + fit + deployed parity exceeded120s; no native qualification'
    del F,X,current,successor
    bank=read(BANK);byseed={e['seed']:e for e in bank['bank']};summary=dict(status='fit_complete',fit=report,new_transition_cameras=0,new_initial_cameras=0)
    def finish():
        summary.update(wall_seconds=time.monotonic()-began,source_frozen=spec['source']==base.source_signature() and all(sha(Path(__file__).with_name(n))==h for n,h in spec['files'].items()),
            assets_unchanged=all(sha(p)==h for p,h in spec['pinned_assets'].items()),model_frozen=sha(DEST/'rbf-model.npz')==report['model_sha256'],no_automatic_promotion=True)
        assert summary['new_transition_cameras']<=96000
        write(DEST/'summary.json',summary);print({k:v for k,v in summary.items() if k not in ('fit','screen8','screen32','fresh')},flush=True)
    if report['learned_decisions']==0:
        summary['status']='acquisition_failure_all_fallback';finish();return
    for size in (8,32):
        entries=[byseed[s] for s in spec['screen8' if size==8 else 'screen32']]
        bank_path=DEST/f'screen{size}-bank.json';write(bank_path,dict(bank,bank=entries))
        native_check(DEST/f'screen{size}',bank_path,DEST/'proposal.npy',
            cached_native_root=DEST/'screen8' if size==32 else None,model_bundle=DEST/'rbf-model.npz',
            practical_later_screen=True,native_seconds=max(1,deadline-time.monotonic()),absolute_deadline=deadline)
        s=read(DEST/f'screen{size}/summary.json');summary[f'screen{size}']=s
        summary[f'screen{size}_runtime_audit']=audit_native(DEST/f'screen{size}',deployed)
        summary['new_transition_cameras']+=s['native']['new_transition_cameras'];summary['new_initial_cameras']+=s['native']['new_initials']
        if s['status']!='complete':summary['status']=f'screen{size}_bounded_partial';finish();return
        baseline=(s['baseline']['own_returns'],s['baseline_more']['later_returns'],s['baseline']['first_incoming_returns'],s['baseline_more']['tail']['2'])
        got=stats(s)
        if size==8 and (any(x<y for x,y in zip(got,baseline)) or got[0]<=baseline[0]):
            summary['status']='rejected_screen8';finish();return
    gate=s['conditional_fresh_eligible'] and got[2]>=baseline[2] and got[3]>=baseline[3] and s['candidate_more']['later_fraction']>=s['baseline_more']['later_fraction']
    summary['fresh_gate']=gate
    if gate and time.monotonic()<deadline-2:
        fresh_run(DEST/'fresh',MODEL,DEST/'screen32',absolute_deadline=deadline)
        f=read(DEST/'fresh/summary.json');summary['fresh']=f;summary['status']='fresh_'+f['status']
        summary['fresh_runtime_audit']=audit_native(DEST/'fresh/candidate',deployed)
        summary['new_transition_cameras']+=f['fresh_transition_cameras'];summary['new_initial_cameras']+=f['initial_cameras']
    else:summary['status']='rejected_screen32_or_no_time'
    finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','run']);parser.add_argument('--approved-spec-sha256');args=parser.parse_args()
    if args.operation=='prepare':prepare()
    else:fit_and_run(args.approved_spec_sha256)
