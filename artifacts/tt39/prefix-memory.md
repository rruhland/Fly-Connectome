# Does the arm learn its next return from the current rally?

This audit separates learning within a point from knowing how to rally with the existing fixed controller. It probes the deployed elastic joint MPC, whose online memory stores short, actually issued motor reference fragments credited by a legal near-arm return. The audit does not add labels from missed shots or test the separate contact-timing refit.

## Result

On four prespecified and physically distinct held opportunities, retaining within-point memory yielded **0 additional legal returns and 0 lost legal returns**. All four incoming offers and all subsequently issued joint commands matched exactly. Two offers were returned by both controllers. Both controllers touched but failed to return the other two. There were no absent offers and no censored pairs.

| Held reply seed | Own returns already earned | Newly available memory rows | Erased | Retained | Different next commands |
| --- | ---: | ---: | --- | --- | ---: |
| 374369 | 2 | 10 | Touch, no return | Touch, no return | 0 |
| 374386 | 2 | 9 | Legal | Legal | 0 |
| 374403 | 5 | 22 | Legal | Legal | 0 |
| 374420 | 10 | 42 | Touch, no return | Touch, no return | 0 |

The retained arm's control proposals during the forced prefix reproduced the original commands exactly on every case, and its next search notes also matched. On seed 374369 the erased arm was replayed anew with matching tracker-state, encoder, reference, command and RNG prefix digests, and identical next physical events. The other erased endpoints are the archived physical runs. This demonstrates no adaptation from the existing reward-only fragment rule on these four opportunities. It does not rule out effects on other approaches or with a different memory rule.

## Design

- The primary source is the archived frozen held-reply deck. In recorded deck order, take each distinct physical tape that reached at least two actual near-arm legal returns. Choose its deepest reached boundary among 2, 5, and 10. Hash the complete physical preboundary referee history to exclude any duplicated trajectory. This yields four different reply seeds of the same measured opening, not four independent openings. This range of later replies comes from the live opponent's continuous random draw. The original launch is never counted as an own return.
- Replay the recorded physical commands through the chosen legal award. Run the normal frozen planner and tracker even during that forced prefix, retaining its original warm plans, previous issued reference, camera history, and search RNG. Check that the unforced frozen planner's commands agree with the tape and that the referee event prefix matches exactly.
- The treatment reconstructs only the existing online memory rule using the prefix's observed stereo state, joint encoders, own actually issued references, and legally awarded score. Those reconstructed rows are not readable by the planner until the branch. The control is the archived frozen physical continuation, with separate native replay parity recorded where indicated. Neither arm updates memory after handoff.
- Score exactly one subsequent receiving opportunity. “Legal” requires actual racket contact and a legal far-side landing. “Touch, no return” requires actual near racket contact on the offer without a legal return. “Miss” requires an actual legal far return with no next near contact. “No offer” is distinct from a miss and is excluded from the common-offer comparison. We compare exact incoming legal bounce events, not commanded opponent goals.
- This is a **forced-prefix, conditional intervention**, not a comparison of how often a learning or unlearning arm will reach the prefix on its own. The main cohort is conditional on a pre-existing frozen controller getting to the stated score on four fresh reply streams. It is not a broad estimate of learning from zero experience or a claim across held opening families.

## Supplemental archived counterexample, not pooled

The known novel opening 4640 is a separate, retrospectively selected case. The frozen and active tapes have identical starts and issued actions through the second own award at tick 241. They first differ at issued tick 275. Both physically scored the immediately following third own return. The active return departed at 5.16 m/s and landed at y = 0.496 m, versus 5.99 m/s and y = 1.143 m frozen. The complete points lasted 27 and 53 alternating legal strokes, respectively. The length gap does not mean the immediate next return failed. Credit can alter the ensuing state even when the binary legal score ties. Higher-score prefixes are already different and are not counted as more matched opportunities. The source hashes and both third physical strikes are in [prefix-memory-4640.json](prefix-memory-4640.json).

## Next ablation worth running

Keep the incumbent search candidate list and its random draws identical in both arms. Add remembered movements in a separate small proposal pool so retrieval cannot silently remove a fixed demonstration or reorganize the exploration budget. Log whether a remembered movement was retrieved, checked, actually issued, and followed by a visible hit, a nonlegal contact, or a miss. Promotion should depend on measured performance relative to the incumbent on nearby approaches, with signed credit for both hits and misses. Re-run the forced-prefix comparison before enabling unrestricted substitutions. This isolates whether the credit itself is useful from changes to the search process that merely follow from adding memory rows.

## Reproduction

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1 PYTHONPATH=.:src \
  /workspace/Fly-Connectome/.venv/bin/python -m scripts.experimental.tt39_prefix_memory \
  artifacts/tt39/prefix-memory-held.json.gz
```

Pass `--fresh-control` to freshly replay the erased continuation in every pair rather than the first pair only. The archive hashes, command-prefix binary digests, reconstructed memory digests, sensor prefix hashes, and posthandoff searches are included in [prefix-memory-held.json.gz](prefix-memory-held.json.gz).
