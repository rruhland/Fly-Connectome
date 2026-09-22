"""Finite-history feasibility only; exact event clipping and relaxed quiet cap."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, diags, eye, hstack, vstack

from credit_capacity import design, score
from fly_connectome.data import checksum


def feasibility(x, y, tempo, maximum, anticipation=.1, quiet_limit=.05, time_limit=15.):
    """An infeasible result rules out the stricter actual alarm convention too.

    Quiet flags permit up to floor(.05*N) unconstrained quiet rows per tempo.
    Unflagged rows allow |raw| <= .100001, slightly MORE permissive than the
    model's alarm at |prediction| >= .1. Event variables are bounded above by
    the exact clipped signed forecast. Their required means ensure existence
    iff the encoded forecasts can satisfy the event constraints.
    """
    active = np.any(x != 0, axis=0)
    matrix = x[:, active];k = matrix.shape[1]
    quiet = y == 0;events = ~quiet
    q = csr_matrix(matrix[quiet]);s = csr_matrix(matrix[events]*y[events, None])
    nq, ne = q.shape[0], s.shape[0]
    def zero(rows, cols):
        return csr_matrix((rows, cols))
    qlo = maximum*np.minimum(q.toarray(), 0).sum(1)
    qhi = maximum*np.maximum(q.toarray(), 0).sum(1)
    elo = maximum*np.minimum(s.toarray(), 0).sum(1)
    ehi = maximum*np.maximum(s.toarray(), 0).sum(1)
    threshold = .100001
    blocks = [
        hstack([q, -diags(np.maximum(0, qhi-threshold)), zero(nq, ne), zero(nq, ne)]),
        hstack([-q, -diags(np.maximum(0, -qlo-threshold)), zero(nq, ne), zero(nq, ne)]),
        hstack([s, zero(ne, nq), diags(ehi+1), zero(ne, ne)]),
        hstack([-s, zero(ne, nq), -diags(np.maximum(0, -1-elo)), eye(ne)]),
        hstack([zero(ne, k), zero(ne, nq), 2*eye(ne), eye(ne)])]
    upper = [np.full(nq, threshold), np.full(nq, threshold), ehi, np.zeros(ne), np.ones(ne)]
    lower = [np.full(len(v), -np.inf) for v in upper]
    for t in np.unique(tempo):
        qm = (tempo[quiet] == t).astype(float)
        blocks.append(hstack([zero(1, k), csr_matrix(qm[None]), zero(1, ne), zero(1, ne)]))
        lower.append(np.array([-np.inf]));upper.append(np.array([np.floor(quiet_limit*qm.sum())]))
        for polarity in [-1, 1]:
            em = ((tempo[events] == t) & (y[events] == polarity)).astype(float)
            assert em.any()
            blocks.append(hstack([zero(1, k), zero(1, nq), zero(1, ne), csr_matrix(em[None])]))
            lower.append(np.array([anticipation*em.sum()]));upper.append(np.array([np.inf]))
    count = k+nq+2*ne
    constraints = LinearConstraint(vstack(blocks).tocsr(), np.concatenate(lower), np.concatenate(upper))
    result = milp(np.zeros(count), integrality=np.r_[np.zeros(k), np.ones(nq+ne), np.zeros(ne)],
        bounds=Bounds(np.r_[np.zeros(k+nq+ne), -np.ones(ne)],
                      np.r_[np.full(k, maximum), np.ones(nq+2*ne)]),
        constraints=constraints,
        # HiGHS presolve incorrectly rejects the saved known-feasible multiscale
        # control. Keep this disabled; regression fixture preserves that case.
        options={'time_limit': time_limit, 'mip_rel_gap': 0., 'presolve': False})
    weights = None
    violation = None
    if result.x is not None:
        lhs = constraints.A@result.x
        violation = float(max(0., np.max(constraints.lb-lhs), np.max(lhs-constraints.ub)))
        assert violation <= 1e-6
        weights = np.zeros(x.shape[1]);weights[active] = result.x[:k]
    return weights, dict(status=int(result.status), message=result.message,
        quiet_relaxation_threshold=threshold, active_columns=np.flatnonzero(active).tolist(),
        variables=count, binary_variables=nq+ne, time_limit_seconds=time_limit,
        presolve=False, constraint_violation=violation)


def main():
    root = Path('runs/credit-capacity-v1')
    a = np.load(root/'predictions.npz')
    old = json.loads((root/'results.json').read_text())
    assert checksum(root/'predictions.npz') == old['artifacts'][str(root/'predictions.npz')]
    train = a['train'];output = {}
    cases = [('shared', False, train, .05), ('split', True, train, .05)]
    cases += [(f'split-tempo{d}', True, train & (a['tempo'] == d), .05) for d in [2, 4, 6]]
    cases += [('split-no-quiet-control', True, train, 1.)]
    for name, split, fitting, quiet_limit in cases:
        x = design(a['x'], a['state'], split)
        weights, info = feasibility(x[fitting], a['y'][fitting], a['tempo'][fitting], old['maximum_weight'], quiet_limit=quiet_limit)
        if name == 'split-no-quiet-control':
            assert info['status'] == 0
        if weights is not None:
            info['coefficients'] = weights.tolist()
            prediction = x@weights
            info['training_scores'] = {str(d): score(a['y'][mask], prediction[mask])
                for d in np.unique(a['tempo'][fitting]) if (mask := fitting & (a['tempo'] == d)).any()}
            info['scores'] = {f'test{d}': score(a['y'][mask], (x@weights)[mask])
                for d in [2, 3, 4, 6] if (mask := a['test'] & (a['tempo'] == d)).any()}
        output[name] = info
        print(name, json.dumps(info), flush=True)
        (root/'feasibility.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    (root/'feasibility.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
