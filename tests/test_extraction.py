import numpy as np
import pytest

from fly_connectome.extraction import Selection, extract_tables


def tables():
    types = ['L1', 'Mi1', 'T4a', 'LC10a', 'central1', 'DNa02', 'DNa02', 'L5']
    ids = list(range(10, 90, 10))
    rows = [dict(bodyId=b, type=t, somaSide='R' if b != 70 else 'L', status='Traced',
                 superclass='descending_neuron' if t == 'DNa02' else ('cb_intrinsic' if b == 50 else 'ol_intrinsic'),
                 assignedOlHex1=0., assignedOlHex2=0., somaLocation=[b, 0, 0]) for b, t in zip(ids, types)]
    nt = [dict(body=b, predicted_nt='acetylcholine', predicted_nt_confidence=.9) for b in ids]
    rois = {b: {r: {'pre': 5} for r in ('LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AOTU(R)', 'PVLP(R)', 'PLP(R)')} for b in ids}
    edges = np.array([[10,20,5],[20,30,5],[30,40,5],[40,50,5],[50,60,5],[50,70,5],[80,20,5]])
    return rows, nt, rois, edges


def test_extraction_preserves_induced_edges_and_excludes_deferred_types():
    result = extract_tables(*tables(), Selection(max_hops=6, required_types=('L1', 'T4a', 'LC10a', 'DNa02')))
    assert result.graph.body_ids.tolist() == [10,20,30,40,50,60,70]
    assert len(result.graph.pre) == 6
    assert result.threshold == 5
    assert result.motor_up == [60] and result.motor_down == [70]
    assert result.pathways == ['feedforward']*3 + ['behavioral']*3
    assert result.retina['neuron_columns'] == [0,-1,-1,-1,-1,-1,-1]


def test_missing_preregistered_population_fails_before_training():
    with pytest.raises(ValueError, match='required population'):
        extract_tables(*tables(), Selection(required_types=('L2',)))


def test_curated_transmitter_evidence_is_used_with_provenance():
    rows, nt, rois, edges = tables()
    nt[0].update(predicted_nt='unclear', ground_truth='glutamate', consensus_nt='glutamate')
    result = extract_tables(rows, nt, rois, edges, Selection(max_hops=6, required_types=('L1',)))
    assert result.graph.signs[0] == -1
    assert result.sign_evidence[0]['source'] == 'ground_truth'


def test_untyped_central_neurons_are_retained_on_measured_paths():
    rows, nt, rois, edges = tables()
    rows[4]['type'] = None
    result = extract_tables(rows, nt, rois, edges, Selection(max_hops=6, required_types=('L1', 'T4')))
    assert 50 in result.graph.body_ids
    assert result.retina['cell_types'][4] == 'untyped'
