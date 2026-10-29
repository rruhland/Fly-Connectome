import numpy as np
from public_entity_pong_native import snapshot,save_public_entities,load_public_entities
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy,PersistentRelationalController


def entities():
    return [dict(id=i,normalized_position=np.array([y,x],np.float32),observed_velocity=np.array([.1,0],np.float32),
        observed=True,velocity_known=True,observation_age_seconds=0.,velocity_age_seconds=None if i==11 else 0.,
        association_strength=.75) for i,y,x in ((11,.3,.1),(23,.6,.5),(51,.2,.9))]


def test_public_archive_roundtrip_preserves_identity_none_age_and_owned_values(tmp_path):
    source=entities();first=snapshot(source);source[0]['normalized_position'][0]=.9
    assert first[0]['normalized_position'][0]!=float(source[0]['normalized_position'][0])
    second=snapshot(entities());second[1].update(observed=False,observation_age_seconds=.02,velocity_age_seconds=.02)
    p=tmp_path/'public.npz';save_public_entities(p,[[],first,second])
    restored=load_public_entities(p)
    assert restored==[[],first,second]
    assert restored[1][0]['velocity_age_seconds'] is None and restored[1][1]['velocity_age_seconds']==0.
    assert restored[2][1]['id']==23 and not restored[2][1]['observed']


def test_serialized_public_inputs_replay_bound_ids_and_actual_command_history(tmp_path):
    first=snapshot(entities());partial=snapshot(entities());partial[1].update(observed=False,
        observation_age_seconds=.04,velocity_age_seconds=.04)
    p=tmp_path/'public.npz';save_public_entities(p,[first,partial,first])
    original=RelationalPolicy(3,0,1,0,-1,np.zeros(5),np.ones(4))
    for gate in (False,True):
        policy=PersistentRelationalPolicy(original,gate);live=PersistentRelationalController(policy);replay=PersistentRelationalController(policy)
        for index,(a,b) in enumerate(zip([first,partial,first],load_public_entities(p))):
            action,decision=live.choose(a,2);other,got=replay.choose(b,2)
            assert action==other and decision==got
            live.observe(index==0);replay.observe(index==0)
            if index==1:
                assert decision['binding_mode']=='persistent_visual_carry'
                assert decision['bound_ids']==[11,23] and not decision['selected_entities'][1]['observed']
