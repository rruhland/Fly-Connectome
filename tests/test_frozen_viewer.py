"""Viewer isolation and exact native-game parity; no training is launched."""
import json
from pathlib import Path
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
import pytest
import torch
from fly_connectome.pong import Pong,Physics
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.viewer.catalog import Checkpoint,load_catalog
from fly_connectome.viewer.engine import ViewerSession,HumanPong,vision_overlay


@pytest.fixture(scope='module')
def catalog():
    torch.set_num_threads(1)
    # Tests require no external learned artifact: a deterministic frozen fixture
    # exercises the same evaluator/vision path. Artifact identity is checked below
    # when the experiment bundle is installed.
    rng=np.random.default_rng(17231);weights=rng.normal(0,.01,(3,4389));weights.setflags(write=False)
    return {'final':Checkpoint('final','Frozen fixture','TEST','Test only',weights,None,'fixture',{})}


@pytest.mark.parametrize('seed',[123456789,2047868701])
def test_watch_exact_native_records_and_terminal_state(catalog,seed):
    viewer=ViewerSession(catalog,seed=seed)
    ref=FreshPongEvaluator(catalog['final'].weights,PongEvaluationConfig(seed=seed,episodes=1000000,max_episode_ticks=2400,opponent_speed=.65))
    ref._begin_episode();viewer.playing=True
    for _ in range(1000):
        actual=viewer.advance()[0];expected=ref.step(check_frozen=False)
        assert actual==expected
        packet=viewer.packet()
        assert packet['camera']['on_pixels']==ref.previous.flatten().nonzero().flatten().tolist()
        assert packet['observations']['transitions']==ref.transitions
        if expected['terminated'] or expected['truncated']:break
    else:pytest.fail('ordinary game failed to end within its native cap')
    assert fingerprint(viewer.evaluator)==fingerprint(ref)
    viewer.evaluator.verify_frozen();ref.verify_frozen()


def test_overlay_read_is_pure_and_uses_sensor_fields(catalog):
    viewer=ViewerSession(catalog);viewer.playing=True
    for _ in range(24):viewer.advance()
    session=viewer.evaluator;before=fingerprint(session)
    result=vision_overlay(session.latest_encoded)
    assert result['entities'] and result['forecasts']
    assert {f['horizon'] for f in result['forecasts']}<= {1,4,8}
    for _ in range(3):viewer.packet()
    assert fingerprint(session)==before
    # Presentation output owns ordinary Python values, not mutable model tensors.
    result['entities'][0]['position'][0]=999
    assert fingerprint(session)==before


def test_human_physics_matches_native_given_identical_right_controller():
    native=Pong([15317],config=Physics(opponent_speed=.65));human=HumanPong([15317],config=native.config)
    for i in range(100):
        desired=float(((native.ball[:,1]-native.opponent)*4).clamp(-.65,.65)[0]);human.human_axis=desired/.65
        drive=torch.tensor([(-1.,0.,1.)[i%3]])
        a=native.step(drive,auto_reset=False);b=human.step(drive,auto_reset=False)
        for field in ('ball','ball_velocity','opponent','opponent_velocity','rally_steps','rng'):
            assert torch.equal(getattr(native,field),getattr(human,field)),field
        for field in ('position','activation','velocity'):assert torch.equal(getattr(native.body,field),getattr(human.body,field))
        assert torch.equal(a.scores,b.scores) and torch.equal(a.hits,b.hits)


def test_human_drive_and_request_only_clock(catalog):
    clock=[100.];viewer=ViewerSession(catalog,mode='human',clock=lambda:clock[0]);viewer.playing=True
    initial=float(viewer.evaluator.env.game.opponent[0]);viewer.advance(human_axis=1.)
    assert float(viewer.evaluator.env.game.opponent[0])>initial
    n=viewer.evaluator.transitions
    with pytest.raises(ValueError):viewer.advance(frames=2)
    clock[0]+=10
    assert viewer.packet()['playing'] is False
    assert viewer.evaluator.transitions==n  # No requests, no background game time.
    assert viewer.advance()==[] and viewer.evaluator.transitions==n
    viewer.advance(manual=True,human_axis=-1.)
    assert viewer.evaluator.transitions==n+1 and not viewer.playing
    viewer.evaluator.verify_frozen()


def test_reset_reproduces_game_and_invalid_reset_preserves_state(catalog):
    viewer=ViewerSession(catalog);viewer.playing=True
    a=[viewer.advance()[0] for _ in range(12)]
    viewer.reset('final',123456789,'watch');viewer.playing=True
    assert a==[viewer.advance()[0] for _ in range(12)]
    before=fingerprint(viewer.evaluator)
    with pytest.raises(ValueError):viewer.reset('unknown',123,'watch')
    assert fingerprint(viewer.evaluator)==before


def test_http_has_no_training_or_checkpoint_write_endpoints(monkeypatch,catalog):
    from fly_connectome.viewer import server
    monkeypatch.setattr(server,'load_catalog',lambda *args:(catalog,[]))
    http=server.make_server('.',port=0);thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start();root=f'http://127.0.0.1:{http.server_port}'
    def post(path,data):return json.load(urlopen(Request(root+path,json.dumps(data).encode(),{'Content-Type':'application/json'})))
    try:
        assert b'Learning you can play against' in urlopen(root).read()
        response=post('/api/session',{});identity=response['session'];revision=response['state']['revision']
        for action in ('train','save','start_worker','restart'):
            with pytest.raises(HTTPError) as error:post('/api/command',dict(session=identity,action=action))
            assert error.value.code==400
        for route in ('/start','/command','/api/train'):
            with pytest.raises(HTTPError) as error:post(route,{})
            assert error.value.code==404
        state=post('/api/command',dict(session=identity,action='step',revision=revision,frames=1))['state']
        assert state['observations']['transitions']==1 and state['frozen']
    finally:http.shutdown();http.server_close()


def test_installed_checkpoint_bundle_is_readonly_and_exact():
    root=Path('/workspace/m1b-experiments');bundle=root/'pong-interception-checkpoint-20261025'
    if not bundle.exists():pytest.skip('optional installed research artifacts absent')
    catalog,missing=load_catalog(root);assert not missing
    assert catalog['final'].digest=='184dd67a67999224dbe4eb5ff3ba3b86ebdf821d15f1486eef2432a56e99fb15'
    assert np.array_equal(catalog['final'].weights,np.load(bundle/'weights.npy'))
    assert not catalog['final'].weights.flags.writeable
    assert catalog['final'].evidence['metrics'][1]['hits']==117
    # Same installed prior/config metadata was checked for all historical stages.
    for checkpoint in catalog.values():assert checkpoint.weights.shape==(3,4389)
