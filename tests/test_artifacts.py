import json
import numpy as np

from test_extraction import tables
from fly_connectome.extraction import Selection, extract_tables
from fly_connectome.artifacts import initialize


def test_visual_stage_and_warm_motor_expansion(tmp_path):
    rules = Selection(max_hops=6, required_types=('L1',))
    extracted = extract_tables(*tables(), rules)
    g = extracted.graph
    np.savez(tmp_path / 'graph-t1.npz', body_ids=g.body_ids, pre=g.pre, post=g.post,
             contacts=g.contacts, signs=g.signs, gain=g.gain)
    manifest = dict(dataset='test-fixture', graph_sha256=g.identity(), threshold=5,
        pathways=extracted.pathways, motor_up=extracted.motor_up, motor_down=extracted.motor_down,
        retina=extracted.retina, positions=extracted.positions, regions=extracted.regions,
        stages=extracted.stages)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    a = initialize(tmp_path, stage='M1A', seeds=[1])
    assert a.network.graph.body_ids.tolist() == [10,20,30,40]
    assert not (a.network.pathways == 2).any()
    a.network.magnitudes.fill_(.03)
    path = tmp_path/'visual.pt'
    a.save(path)
    b = initialize(tmp_path, stage='M1B', seeds=[1], warm_checkpoint=path)
    assert b.network.graph.body_ids.tolist() == [10,20,30,40,50,60,70]
    assert b.network.magnitudes[:3].tolist() == a.network.magnitudes.tolist()
    np.testing.assert_allclose(b.network.magnitudes[3:].numpy(), [.025]*3, atol=1e-8)
