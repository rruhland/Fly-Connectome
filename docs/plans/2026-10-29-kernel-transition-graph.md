# One offline anonymous transition-graph prototype

Authorized as a parallel offline comparison while the paired forest runs. No
native games are authorized by this document. All implementation lives in new
`pong_kernel_transition_graph.py`, `run_kernel_transition_graph.py` and its test
file; existing shared helpers stay unchanged.

Use every eligible transition in the 60 continuous-actor TRAIN traces. The archive
has 27,479 source cameras and 118 own hits, all hits on interior rows. Retain every
interior row with its actual successor. Additionally retain a final row only when
the receipt verifies natural termination: its zero bootstrap requires no successor.
Exclude censored final rows with unavailable successors. No reward oversampling,
fresh data, expert labels or gradients. Start IDs establish independent local
support and never enter runtime state.

Represent the complete anonymous scene with at most four six-field rows. Positions
are `(y,x)`, so coordinate1 is the primary sorting key. Pad continuous coordinates
to 16 dimensions, divide each entity's four continuous fields by the frozen
2,048-scene calibration deviations with the existing floor, then divide the full
vector by four. Exact discrete strata comprise entity presence, observed and
velocity-known masks, past observed own-hit reward, and public camera ordinal mod 5.
There are no baseline scores in graph features. Overflow is unsupported.
Training subsequently proved to contain 65 five-entity scenes despite calibration's
maximum of 4. Retain those transitions in the dataset and their reward evidence;
their source states cannot supply neighbors, and their successor states form
explicit unsupported model boundaries. Capacity does not expand.

The one radius is the 95th percentile Euclidean distance between same-mask,
same-phase calibration scenes five cameras apart. It is selected without rewards.
For each action and stratum, take the nearest eight source rows within that radius;
support requires at least two distinct TRAIN starts among them. Normalize positive
Gaussian weights `exp(-.5*(distance/radius)**2)` to sum to one. Unsupported action
weights are zero. No radius, feature, support or neighbor-count sweep follows.

Each transition value is one for an own hit; zero for a verified natural end
without a hit; otherwise it is its actual elapsed discount multiplied by the
largest supported action value at its saved successor. A successor with no action
support contributes zero as an explicitly identified MODEL BOUNDARY, not a
natural-miss training observation. The target concerns the next own hit, so hits
are absorbing success for this prediction. `discount=2**(-elapsed_seconds/3.2)`
sets a half-life of 160 normal 20ms camera periods. The public sensor scheduler
alternates 2/3 ticks according to `floor((i+1)*12/5)-floor(i*12/5)`; reconstruct
interior durations and the final duration against every receipt's elapsed total,
without clipping or inventing terminal timing. The 1/120-second scheduler tick is
timing metadata, never a hidden physics input or geometry feature.

Nonnegative normalized interpolation followed by a supported maximum gives a
discounted contraction. This establishes stability of this approximate model,
not accurate visual state, action coverage, a safe policy, or Pong performance.
The baseline supplies deployment actions only: baseline until an actual own hit,
then require baseline and selected alternative support and a strict predicted
value gain; decide every camera. The implementation provides inspection/query
logic but does not launch a native evaluator.

Implementation cap: 20 minutes. One fixed 60-second offline process allocation includes
data preparation, hashing, neighbors, one fit, support reports and serialization
parity. One thread; at most 4,096 iterations; tolerance 1e-8; stop at the original
deadline. Zero post-hit states with both baseline and an alternative supported is
the pinned negligible-coverage stop. Failure to converge stops the prototype.
The artifact directory cannot be reused for an automatic restart. Report natural
terminals separately from model boundaries, exact included rows/hits, support,
convergence, elapsed time and saved-source decision changes. These source queries
are training diagnostics and establish no fresh improvement. New artifacts must
remain below 50 MB. No native screen before parent review.

The mechanism reuses actual successor observations and all available reward events
rather than transferring a few rescue labels. It also avoids the coarse leaves
that falsely admitted nearly every action in the earlier fixed-partition model.
However, this still relies on geometric proximity. Mask/phase conditioning may
leave little genuine coverage; loosening it could recreate that failure.

Primary sources read:

- [Ormoneit and Sen, Kernel-Based Reinforcement Learning (2002)](https://kylewoodward.com/blog-data/pdfs/references/ormoneit%2Bsen-machine-learning-2002A.pdf):
  normalized positive local averaging and approximate Bellman equations. Its
  asymptotic assumptions do not establish finite-data improvement for these
  correlated policy trajectories.
- [Moore and Atkeson, Prioritized Sweeping (1992)](https://proceedings.neurips.cc/paper_files/paper/1992/file/55743cc0393b1cb4b8b37d09ae48d097-Paper.pdf):
  transition memory and propagation of observed consequences. The prototype uses
  vectorized fixed-point sweeps, not their priority queue or maze performance claim.
- [Atkeson, Nonparametric Model-Based Reinforcement Learning (1997)](https://papers.neurips.cc/paper_files/paper/1997/file/97275a23ca44226c9964043c8462be96-Paper.pdf):
  a separate locally learned dynamics route, including model-error amplification.
  Its demonstrated robot setup supplied identified coordinates and demonstrations.
- [Blundell et al., Model-Free Episodic Control (2016)](https://arxiv.org/html/1606.04460):
  the random-projection branch is no-backprop, but max-ever returns are risk-seeking
  under stochasticity and the five tested Atari games exclude Pong. This prototype
  averages transition evidence instead of storing maximum observed returns.

## Offline outcome — converged, native untested

Two initialization failures preceded any learned fit. The first stopped at 0.46 s
because the shared interpreter selected another checkout; the corrected launch
preserved the original 60-second clock and stopped at 43.11 s when it encountered
five-entity training scenes. Both receipts and source/spec hashes remain in
`/workspace/m1b-experiments/pong-kernel-transition-graph-20261029-r1`. Capacity
stays 4; the targeted overflow correction retains those rows as unsupported.
Parent explicitly authorized a separate 60-second recovery allocation for the first
actual fit; its spec links all prior failure receipts and source changes.

Recovery artifact:
`/workspace/m1b-experiments/pong-kernel-transition-graph-20261029-r1-recovery`.
Seven focused unit checks passed. All 27,419 interiors plus 52 verified natural
terminal finals were included: 27,471 transitions, 118 hits, 8 censored finals
excluded. The 65 source overflow cases and 65 successor overflow cases lose 0 hits
and 0 natural terminal events. All 60 elapsed totals reconstruct within 2.665e-13
seconds; the scheduler source hash matches the archive. Graph fitting converged
after 1,012 iterations at residual 9.966e-9, with contraction-based value-error
bound 5.526e-6. Whole recovery process: 13.56 seconds; 64 serialization/query parity
cases passed; 0 native cameras; 1.21 MB new artifacts. Graph SHA:
`4067f033ec767fa072ab4e9342d0d95b9f27b030cbcda536e16261fd7bdccb75`.

Saved-source post-hit baseline-plus-alternative support is 20,080/22,233 (90.316%).
There are 674 unsupported successor boundaries among 27,301 continuation rows.
The model changes 13,343 saved-source decisions. The fixed calibration radius is
0.7568188 in standardized RMS distance: it admits broad neighborhoods. These figures
show numerical feasibility and the declared support checks, not sufficient causal
coverage or learned performance. In particular, a broad radius can recreate the
earlier coarse-estimator problem despite distinct-start counts. Parent review is
required before any native evaluation; no radius adjustment or second fit follows
from this outcome.

Parent subsequently approved one frozen native comparison, assigned to the
separate evaluation worker. Before its launch, one 0.67-second geometry diagnostic
was saved as `geometry-diagnostic.json` in the recovery artifact. For the 13,343
changed selected actions, nearest distance/radius has median 0.440 and p90 0.723;
nearest other-start distance/radius has median 0.460 and p90 0.760. Eighth distance
has median 0.600 and p90 0.881 among the 10,903 cases with eight neighbors.
Independent-start count has median 5, p10 3, p90 6, range 2–8. The baseline has
an exact self match in 96.7% of eligible queries; its nearest other-start distance
has median 0.250. Alternatives therefore tend to use more distant support than
the behavior action. Only 14/13,343 changes (0.1049%) have action-value gaps at most
twice the numerical error bound (1.1052e-5). Median gap is 0.003906 and p90 is
0.02502. Solver error alone does not explain most changes; these geometric
statistics still do not establish counterfactual accuracy. No new fit or native
cameras were used for this diagnostic, and the decision rule remains fixed.
