import numpy as np
import pytest

from scripts.experimental import tt21_carry_checkpoint as carry
from scripts.experimental.tt3_pong import state_sha256
from scripts.experimental.tt4_transfer_trial import _hash, _write


def test_bundle_rejects_changed_legacy_payload(tmp_path):
    path = tmp_path/'player.json.gz'
    legacy = dict(schema='older', models=[{'gantry': 1}], numeric={'pong': 2})
    saved = dict(schema=carry.SCHEMA, inherited_state_sha256=state_sha256(legacy))
    _write(path, dict(legacy, **{carry.ADDRESS: saved}))
    assert carry.address(path) == saved

    legacy['numeric']['pong'] = 3
    _write(path, dict(legacy, **{carry.ADDRESS: saved}))
    with pytest.raises(ValueError, match='older checkpoint fields changed'):
        carry.address(path)


def test_drive_uses_the_portable_asset_root_and_checks_the_catalogue(tmp_path, monkeypatch):
    atlas = tmp_path/'public-catalogue.npz'
    np.savez_compressed(atlas, tapes=np.zeros((1, 20, 5)))
    legacy = dict(schema='older')
    saved = dict(schema=carry.SCHEMA, inherited_state_sha256=state_sha256(legacy),
                 atlas=dict(path=atlas.name, sha256=_hash(atlas)))
    path = tmp_path/'player.json.gz'
    _write(path, dict(legacy, **{carry.ADDRESS: saved}))

    class Session:
        def __init__(self, manifest, catalogue, eye, *, adaptive):
            assert manifest == saved and eye == 'wrist' and adaptive
            assert catalogue['tapes'].shape == (1, 20, 5)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(carry, 'DriveSession', Session)
    with carry.resume_drive(path, 'wrist', assets=tmp_path) as session:
        assert not session.closed
    assert session.closed

    np.savez_compressed(atlas, tapes=np.ones((1, 20, 5)))
    with pytest.raises(ValueError, match='dry catalogue differs'):
        with carry.resume_drive(path, 'wrist', assets=tmp_path):
            pytest.fail('an altered catalogue reached the player')
