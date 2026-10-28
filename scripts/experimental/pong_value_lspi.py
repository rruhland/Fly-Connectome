"""One bounded LSPI calculation on fixed legal features, without autodiff."""
import time
import numpy as np
import scipy.linalg as la


def fit_lspi(F, N, actions, rewards, discounts, weights, initial_policy,
             *, iterations=3, ridge=.01, deadline=None):
    """LSTDQ moments in three action blocks; logits are never value targets."""
    n, d = F.shape
    assert N.shape == F.shape and np.all(weights > 0)
    assert np.isclose(weights.sum(), 1.) and iterations == 3 and ridge == .01
    qdim = 3*d
    gram = np.zeros((qdim, qdim))
    rhs = np.zeros(qdim)
    for a in range(3):
        mask = actions == a
        sl = slice(a*d, (a+1)*d)
        gram[sl, sl] = F[mask].T @ (weights[mask, None]*F[mask])
        rhs[sl] = F[mask].T @ (weights[mask]*rewards[mask])
    policy = np.asarray(initial_policy).copy()
    history = []
    theta = None
    for iteration in range(iterations):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('LSPI deadline before complete fixed iteration count')
        began = time.monotonic()
        A = gram.copy()
        for a in range(3):
            rows = actions == a
            for b in range(3):
                mask = rows & (policy == b) & (discounts != 0)
                if np.any(mask):
                    A[a*d:(a+1)*d, b*d:(b+1)*d] -= (
                        F[mask].T @ ((weights[mask]*discounts[mask])[:, None]*N[mask]))
        A.flat[::qdim+1] += ridge
        anorm = la.norm(A, 1)
        lu, piv = la.lu_factor(A, check_finite=False)
        solution = la.lu_solve((lu, piv), rhs, check_finite=False)
        rcond, info = la.lapack.dgecon(lu, anorm, norm='1')
        assert info == 0 and np.isfinite(solution).all()
        theta = solution.reshape(3, d)
        q = F @ theta.T
        nextq = N @ theta.T
        targets = rewards + discounts*nextq[np.arange(n), policy]
        residual = q[np.arange(n), actions] - targets
        updated = np.argmax(nextq, axis=1)
        history.append(dict(iteration=iteration+1,
            seconds=time.monotonic()-began, reciprocal_condition_estimate=float(rcond),
            regularized_linear_relative_residual=float(np.linalg.norm(A@solution-rhs)/(np.linalg.norm(rhs)+1e-30)),
            bellman_weighted_rms=float(np.sqrt(weights@(residual*residual))),
            maximum_absolute_q=float(np.max(np.abs(q))),
            maximum_absolute_parameter=float(np.max(np.abs(solution))),
            next_policy_changes=int(np.count_nonzero(updated != policy)),
            target_action_counts=np.bincount(policy, minlength=3).tolist()))
        policy = updated
    return theta, history


def map_weights(theta, denominator, B, raw_dimension=4389):
    raw_count = len(B)
    unscaled = theta / denominator[None, :]
    W = unscaled[:, :raw_count] @ B
    R = unscaled[:, raw_count:]
    assert W.shape == (3, raw_dimension) and R.shape == (3, 512)
    return W, R
