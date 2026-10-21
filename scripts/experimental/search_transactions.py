"""Shared complete-population transactions; task fitness remains in each driver."""
import json,time
import numpy as np

def committed_fitness_rows(api,directory,state):
    """Only optimizer-state-committed results count; orphan results never do."""
    n=state['completed_updates'];initial=api.transaction_initial_boundary(directory)
    previous_hash=initial['theta_hash'];rng=initial['next_rng_state'];used=0
    if type(n) is not int or not 0<=n<=api.MAX_UPDATES:raise ValueError('invalid committed update count')
    for local in range(n):
        g=api.FIRST_GENERATION+local;label=f'g{g:02d}'
        definition=json.loads((directory/(label+'-definition.json')).read_text())
        result=json.loads((directory/(label+'-result.json')).read_text())
        theta=np.load(directory/f'theta_at{local+1}.npy',allow_pickle=False)
        if (definition['generation']!=local or definition['global_generation']!=g
                or definition['input_theta_hash']!=previous_hash or definition['input_rng_state']!=rng
                or result['generation']!=local or result['global_generation']!=g
                or result['new_theta_hash']!=api.array_hash(theta)
                or [r['job'] for r in result['rollouts']]!=[j['id'] for j in definition['jobs']]):
            raise ValueError('committed fitness lineage differs')
        if any(type(r['transitions']) is not int or r['transitions']<0 for r in result['rollouts']):
            raise ValueError('invalid committed fitness counters')
        used+=sum(r['transitions'] for r in result['rollouts'])
        previous_hash=result['new_theta_hash'];rng=definition['rng_state_after_draw']
        if definition['diagnostic_jobs'] and local<n-1 and not (directory/(label+'-anchor-result.json')).exists():
            raise ValueError('advanced past unfinished diagnostics')
    last=f'g{api.FIRST_GENERATION+n-1:02d}' if n else None
    pending=state['pending_diagnostics']
    if (state['next_global_generation']!=api.FIRST_GENERATION+n or state['theta_file']!=f'theta_at{n}.npy'
            or state['theta_hash']!=previous_hash or state['rng_state']!=rng
            or pending not in (None,last) or (not n and pending is not None)):
        raise ValueError('optimizer boundary differs from committed fitness')
    if n:
        definition=json.loads((directory/(last+'-definition.json')).read_text())
        if (pending and not definition['diagnostic_jobs']) or (
                definition['diagnostic_jobs'] and pending is None and not (directory/(last+'-anchor-result.json')).exists()):
            raise ValueError('optimizer diagnostic phase differs')
    return used


def freeze_fields(api,directory,state):
    used=committed_fitness_rows(api,directory,state);g=state['next_global_generation']
    path=directory/f'g{g:02d}-definition.json';draw=json.loads(path.read_text()) if path.exists() else None
    if draw and (draw['generation']!=state['completed_updates'] or draw['global_generation']!=g
            or draw['input_rng_state']!=state['rng_state'] or draw['input_theta_hash']!=state['theta_hash']):
        raise ValueError('pending transition draw differs')
    if (directory/f'g{g:02d}-result.json').exists():raise ValueError('unreconciled fitness result at frozen boundary')
    current=api.counts(directory,json.loads((directory/'manifest.json').read_text()))
    if not 0<=used<=current['training_fitness'] or current['training_fitness']+current['training_anchor']>api.TRAIN_CAMERAS:
        raise ValueError('training camera counters differ')
    fitness_failed,anchor_failed=api.logged_failures(directory)
    return dict(new_updates=state['completed_updates'],cumulative_learned_updates=api.INHERITED_UPDATES+state['completed_updates'],
        theta_file=state['theta_file'],theta_hash=state['theta_hash'],fitness_cameras=current['training_fitness'],
        used_fitness_cameras=used,unused_partial_fitness_cameras=current['training_fitness']-used,
        diagnostic_cameras=current['training_anchor'],
        total_training_phase_cameras=current['training_fitness']+current['training_anchor'],
        pending_diagnostics=state['pending_diagnostics'],anchor_failure=anchor_failed,fitness_failure=fitness_failed,
        diagnostic_status='failed' if anchor_failed else 'partial' if state['pending_diagnostics'] else 'complete',
        unused_draw_skipped=bool(draw),next_global_generation=g+int(bool(draw)),
        next_rng_state=draw['rng_state_after_draw'] if draw else state['rng_state'],
        valid=state['completed_updates']>=api.MIN_UPDATES and not fitness_failed)


def train(api,directory,pool,manifest,z,state,theta):
    marker=directory/'training-endpoint.json'
    if marker.exists():
        frozen=json.loads(marker.read_text())
        if any(frozen[k]!=v for k,v in freeze_fields(api,directory,state).items()):
            raise ValueError('frozen endpoint differs from optimizer')
        return frozen,state,theta
    deadline=manifest['started_wall_ns']/1e9+api.TRAIN_SECONDS
    excluded=(*json.loads((directory/'seed-exclusions.json').read_text())['ids'],
        *manifest['serve_seeds'],*manifest.get('action_seeds',()))
    fitness_failed,anchor_failed=api.logged_failures(directory)
    reason='prior logged fitness failure' if fitness_failed else None
    while not fitness_failed:
        if state['pending_diagnostics']:
            label=state['pending_diagnostics'];definition=json.loads((directory/(label+'-definition.json')).read_text())
            if anchor_failed:reason='prior logged diagnostic failure';break
            result=api.execute(directory,definition['diagnostic_jobs'],pool,deadline,api.TRAIN_CAMERAS,api.training_count(directory))
            if not result['complete']:
                anchor_failed=result['failed'];reason='diagnostic failure' if anchor_failed else 'training camera/wall budget during diagnostics';break
            api.finish_diagnostics(directory,definition)
            state=dict(state,pending_diagnostics=None);api.atomic_json(directory/'search-state.json',state)
        if state['completed_updates']>=api.MAX_UPDATES:reason='maximum complete new updates';break
        # Reconcile a complete pinned population before any remaining-budget test.
        # A result-to-state crash consumes no new observations and never drops or
        # double-applies its update, even when the original budget is exhausted.
        pending=directory/f"g{state['next_global_generation']:02d}-definition.json"
        if pending.exists():
            definition,directions=api.prepare_population(directory,z,state,theta,excluded)
            if all(api.completion.complete_report(directory,j) is not None for j in definition['jobs']):
                state,theta=api.finish_fitness(directory,definition,directions,theta,z);continue
        if api.TRAIN_CAMERAS-api.training_count(directory)<200 or deadline-time.time()<=25:
            reason='training camera/wall budget';break
        definition,directions=api.prepare_population(directory,z,state,theta,excluded)
        result=api.execute(directory,definition['jobs'],pool,deadline,api.TRAIN_CAMERAS,api.training_count(directory))
        if not result['complete']:
            fitness_failed=result['failed'];reason='fitness failure' if fitness_failed else 'training camera/wall budget during fitness'
            api.atomic_json(directory/'incomplete-fitness-generation.json',dict(global_generation=definition['global_generation'],**result));break
        state,theta=api.finish_fitness(directory,definition,directions,theta,z)
        print(json.dumps(dict(new_updates=state['completed_updates'],global_generation=definition['global_generation'],
            training_phase_cameras=api.training_count(directory))),flush=True)
    frozen=dict(**freeze_fields(api,directory,state),training_stop=reason,frozen_wall_ns=time.time_ns())
    api.atomic_json(marker,frozen);return frozen,state,theta


