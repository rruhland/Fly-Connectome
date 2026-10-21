"""Complete proposal-cycle commits for a prospectively bounded child selector.

All vision/rollouts stay in the existing camera-durable worker. These helpers
never create observations, choose a checkpoint, or start/reset a wall clock.
"""
import json
import numpy as np


def choose_child(wins,discovery_serves=12):
    """Index0 is incumbent; child ties resolve by their fixed index order."""
    if (len(wins)!=5 or any(type(v) is not int or not 0<=v<=discovery_serves for v in wins)):
        raise ValueError('five complete integer discovery win counts required')
    best=min(range(4),key=lambda i:(-wins[i+1],i))
    return best if wins[best+1]>wins[0] else None


def decision(validation,minimum_net_wins=2):
    if (validation is None or set(validation)!={'incumbent','child','reports'}
            or any(type(validation[k]) is not int or not 0<=validation[k]<=24 for k in ('incumbent','child'))):
        raise ValueError('complete24-serve validation counts required')
    return validation['child']-validation['incumbent']>=minimum_net_wins


def calculate_result(api,directory,definition,state,theta):
    """Pure reconciliation, including source reports and the executed child."""
    actual=api.array_hash(theta)
    if (definition['cycle']!=state['completed_cycles'] or definition['cycle']!=state['next_cycle']
            or definition['input_theta_hash']!=actual or state['theta_hash']!=actual
            or definition['input_rng_state']!=state['rng_state']):
        raise ValueError('stale proposal cycle/theta/RNG cannot be committed')
    discovery=api.discovery_result(directory,definition)
    if discovery is None:return None
    selected=discovery['selected_child']
    validation=None if selected is None else api.validation_result(directory,definition,selected)
    if selected is not None and validation is None:return None
    accepted=selected is not None and decision(validation)
    proposed=(np.load(directory/definition['child_theta_files'][selected],allow_pickle=False)
              if accepted else theta.copy())
    n=state['completed_cycles']+1
    reports=[*discovery['reports'],*(validation['reports'] if validation is not None else [])]
    if any(type(r['transitions']) is not int or r['transitions']<0 for r in reports):
        raise ValueError('valid committed discovery/validation counters required')
    result=dict(cycle=definition['cycle'],input_theta_hash=api.array_hash(theta),discovery=discovery,
        validation=validation,validation_used=selected is not None,accepted=bool(accepted),
        new_theta_file=f'theta_after_cycle{n}.npy',new_theta_hash=api.array_hash(proposed),
        rng_state_after_draw=definition['rng_state_after_draw'],
        committed_training_cameras=sum(r['transitions'] for r in reports),
        completed_cycles=n,accepted_updates=state['accepted_updates']+int(accepted),
        rule='Strict discovery improvement, then fresh24 validation net>=2; fixed ties reject')
    return result,proposed


def finish_cycle(api,directory,definition,state,theta):
    result=calculate_result(api,directory,definition,state,theta)
    if result is None:raise ValueError('partial cycle cannot change the incumbent')
    value,proposed=result
    api.immutable_array(directory/value['new_theta_file'],proposed)
    api.weights_file(directory,f"incumbent_after_cycle{value['completed_cycles']}.npy",
                     proposed@np.load(directory/'search-transform.npy',allow_pickle=False))
    # Result first, state second: an interrupted commit is reconciled, including
    # at an exhausted original budget, without observations or a second update.
    api.immutable_json(directory/f"c{definition['cycle']:02d}-result.json",value)
    new=dict(completed_cycles=value['completed_cycles'],accepted_updates=value['accepted_updates'],
        next_cycle=value['completed_cycles'],theta_file=value['new_theta_file'],theta_hash=value['new_theta_hash'],
        rng_state=value['rng_state_after_draw'])
    api.atomic_json(directory/'search-state.json',new)
    return new,proposed


def committed_rows(api,directory,state):
    initial=api.initial_boundary(directory);previous_file='theta_at0.npy';previous_hash=initial['theta_hash']
    rng=initial['next_rng_state'];accepted=0;used=0;n=state['completed_cycles']
    if type(n) is not int or not 0<=n<=api.MAX_CYCLES:raise ValueError('invalid committed cycle count')
    for c in range(n):
        definition=json.loads((directory/f'c{c:02d}-definition.json').read_text())
        retained=json.loads((directory/f'c{c:02d}-result.json').read_text())
        if (definition['cycle']!=c or definition['input_theta_hash']!=previous_hash
                or definition['input_rng_state']!=rng):raise ValueError('committed cycle lineage differs')
        theta=np.load(directory/previous_file,allow_pickle=False)
        before=dict(completed_cycles=c,accepted_updates=accepted,next_cycle=c,theta_file=previous_file,
            theta_hash=previous_hash,rng_state=rng)
        calculated=calculate_result(api,directory,definition,before,theta)
        if calculated is None or calculated[0]!=retained:raise ValueError('committed decision/raw receipts differ')
        result,proposed=calculated
        if not np.array_equal(proposed,np.load(directory/result['new_theta_file'],allow_pickle=False)):
            raise ValueError('committed incumbent differs from executed decision')
        head=directory/f'incumbent_after_cycle{c+1}.npy'
        mapped=proposed@np.load(directory/'search-transform.npy',allow_pickle=False)
        if (not np.array_equal(mapped,np.load(head,allow_pickle=False))
                or json.loads(head.with_suffix('.json').read_text())!=dict(shape=list(mapped.shape),weight_hash=api.array_hash(mapped))):
            raise ValueError('committed incumbent mapped readout changed')
        used+=result['committed_training_cameras'];accepted+=int(result['accepted'])
        previous_file=result['new_theta_file'];previous_hash=result['new_theta_hash'];rng=result['rng_state_after_draw']
    expected=dict(completed_cycles=n,accepted_updates=accepted,next_cycle=n,theta_file=previous_file,
                  theta_hash=previous_hash,rng_state=rng)
    if state!=expected:raise ValueError('incumbent state/cycle/acceptance/RNG boundary differs')
    return used


def freeze_fields(api,directory,state):
    used=committed_rows(api,directory,state);cycle=state['next_cycle']
    path=directory/f'c{cycle:02d}-definition.json';draw=json.loads(path.read_text()) if path.exists() else None
    if draw and (draw['cycle']!=cycle or draw['input_theta_hash']!=state['theta_hash']
                 or draw['input_rng_state']!=state['rng_state']):raise ValueError('pending cycle draw differs')
    if (directory/f'c{cycle:02d}-result.json').exists():raise ValueError('unreconciled cycle result at freeze')
    observed=api.training_count(directory);failed=api.training_failed(directory)
    if not 0<=used<=observed<=api.TRAIN_CAMERAS:raise ValueError('training cycle camera accounting differs')
    return dict(completed_cycles=state['completed_cycles'],accepted_updates=state['accepted_updates'],
        theta_file=state['theta_file'],theta_hash=state['theta_hash'],used_training_cameras=used,
        unused_partial_cycle_cameras=observed-used,total_training_phase_cameras=observed,
        training_failure=failed,unused_draw_skipped=bool(draw),next_cycle=cycle+int(bool(draw)),
        next_rng_state=draw['rng_state_after_draw'] if draw else state['rng_state'],
        valid=state['completed_cycles']>=api.MIN_CYCLES and not failed)
