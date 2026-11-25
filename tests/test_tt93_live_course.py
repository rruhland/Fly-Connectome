from collections import deque

import numpy as np
import pytest

from scripts.experimental import tt93_live_course as live


def test_two_modes_share_only_one_fully_frozen_opening_and_goal_roster():
    p=live.design()
    cards=p['cards']
    assert len({r['card']['id'] for r in cards})==24
    assert not ({r['card']['id'] for r in cards} & set(p['withheld']))
    assert cards[0]['arm_order']==['stereo','fused']
    assert cards[1]['arm_order']==['fused','stereo']
    for r in cards:
        assert r['card']['opening']=='random' and r['card']['profile']=='reach'
        assert live.digest_json(live.targets(r['card']))==r['goals']


def test_memory_checkpoint_stays_mode_local_and_checks_the_native_prefix(tmp_path,monkeypatch):
    p=tmp_path/'protocol.json';p.write_text('{}')
    monkeypatch.setattr(live,'ROOT',tmp_path)
    monkeypatch.setattr(live,'PLAN',p)
    class Bank:
        def __init__(self):
            self.online=deque(maxlen=768)
        def digest(self):
            return live.digest_json([[np.asarray(x).tolist(),np.asarray(y).tolist(),t] for x,y,t in self.online])
    class Ctl:
        def __init__(self):
            self.priors=Bank()
            self.serial=48
    before=Ctl()
    data=live.border(before,'stereo')
    before.priors.online.append((np.ones(2),np.zeros(6),('live',49,1)))
    before.serial+=1
    live.append(before,data,dict(memory={'after':before.priors.digest()}))
    other=Ctl()
    assert live.border(other,'fused')['rows']==[]
    assert not other.priors.online
    live.border(other,'stereo')
    assert other.priors.digest()==before.priors.digest() and other.serial==49
    f=live.filename('stereo')
    q=live._read(f);q['checkpoint_sha256']='not a signature'
    live._write(f,q)
    with pytest.raises(RuntimeError):
        live.border(Ctl(),'stereo')
