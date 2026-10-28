"""Certify episode reuse under a changed stateless readout, without new cameras.

Caller must retain the same deterministic engine/runtime, visual prior, seed,
configuration, scaler and bookkeeping RNG setup. Source fingerprints and exact
source q records are checked here. Unchanged actions at EVERY cached input imply
unchanged future inputs by induction; checking only successful prefixes is not
sufficient. Bulk sparse matmul is deliberately not used for near-tie decisions.

This returns a certificate and updated q records, not a fabricated execution
receipt. Callers must count reused cameras separately from executed cameras and
retain source receipt/feature/weight provenance.
"""
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.policy import FrozenLinearPolicy


def certify_episode(receipt, features, source_weights, candidate_weights, *,
                    source_weights_sha256, source_fingerprint,
                    candidate_fingerprint, candidate_config,candidate_readout=None):
    def reject(reason, checked=0, index=None):
        return dict(certified=False, reason=reason, checked_rows=checked,
                    first_disagreement=index, executed_transition_cameras=0,
                    executed_initial_cameras=0)
    if not source_fingerprint or source_fingerprint != candidate_fingerprint:
        return reject('engine_or_runtime_context_changed')
    if receipt.get('transport_kind','visual-transport-v1')!='visual-transport-v1':
        return reject('nondefault_transport_requires_explicit_context_certificate')
    config=receipt['config']
    if config != candidate_config:
        return reject('configuration_changed')
    if (config.get('policy_kind')!='trained' or config.get('action_hold_frames')!=1
            or config.get('episodes')!=1 or config.get('action_seed') is not None):
        return reject('unsupported_policy_or_episode_mode')
    if not receipt['complete'] or not receipt['report']['complete']:
        return reject('incomplete_source_episode')
    if receipt['weights_sha256'] != source_weights_sha256:
        return reject('source_weight_hash_mismatch')
    if receipt.get('feature_format')!='actual capped raw CSR':
        return reject('unsupported_feature_provenance')
    if not sp.isspmatrix_csr(features):
        return reject('features_must_be_exact_csr')
    records=receipt['records'];n=len(records)
    if features.shape != (n, np.asarray(source_weights).shape[1]) or n!=receipt['transitions']:
        return reject('incomplete_feature_trajectory')
    if np.asarray(candidate_weights).shape!=np.asarray(source_weights).shape:
        return reject('readout_shape_changed')
    source=FrozenLinearPolicy(source_weights);candidate=FrozenLinearPolicy(candidate_weights) if candidate_readout is None else candidate_readout
    updated=[]
    for i,record in enumerate(records):
        lo,hi=features.indptr[i:i+2];indices=features.indices[lo:hi];values=features.data[lo:hi]
        old_action,old_q=source.choose(indices,values)
        if old_action!=record['action'] or not np.array_equal(old_q,np.asarray(record['q'])):
            return reject('source_policy_record_mismatch',i+1,i)
        action,q=candidate.choose(indices,values)
        if action!=record['action']:
            return reject('candidate_action_changed',i+1,i)
        updated.append(dict(record,q=q.tolist()))
    return dict(certified=True,reason='all_exact_policy_actions_identical',checked_rows=n,
                records=updated,logical_transition_cameras=n,
                reused_initial_cameras=receipt['initial_cameras'],
                executed_transition_cameras=0,executed_initial_cameras=0,
                proof='Same deterministic initial state and actions preserve every subsequent sensor input; exact source and candidate choose used on every row.')
