# Next scaffold target: action-specific reward credit

The user renewed continuous orchestration on 2026-10-22, until told to stop.
Continue bounded cycles under the [general plan](../plans/2026-10-21-continuous-learning-scaffolds.md).
The prior refinement found no incremental learning; its negative result remains
unchanged. This cycle tests a different credit mechanism within the same visual
representation and linear policy class. No backpropagation or privileged policy
inputs are introduced.

## Independent challenge changed the direction

Astra inspected motor dynamics, visual velocity encoding and spatial pooling,
and asked Sol to distinguish missing actuator history from unreliable search
credit. The reused Sol 6.1 XHigh agent challenged its own earlier action-history
recommendation using retained data. Among 30 first misses, 22 were moving toward
the ball, six were nearly stationary, and two were moving away. Both moving-away
cases were still actuated away; none showed the specific already-braking pattern.
This weakens that particular braking explanation, without establishing correct
perception or excluding earlier timing errors.

Only 19 of the 48 training starts across four previous search populations supplied
any mirrored differential reward. Split-half update directions were often opposed
in two populations. These overlapping retrospective comparisons are diagnostics,
not independent significance tests. They justify reconsidering credit quality;
nonzero parameter updates alone did not establish a reliable learning direction.
The diagnosis replayed 5,685 old physics transitions in 7.11s, with zero new
candidate observations. Diagnostic state and labels never enter the learner.

## Selected bounded intervention

Carry the original first-qualified final return head, frozen visual system,
transform and feature cap. Explicitly discard the unsuccessful refinement's four
additional updates for this same-start mechanism comparison. Keep the existing
one-action-per-camera cadence. No action history or recurrent architecture is
added. The policy's outputs remain action preferences, not fitted Q-values.

Four fresh first-hit episodes share fixed weights within each batch. A stochastic
linear actor accumulates an explicit local eligibility:

`p = softmax(theta x / temperature)`

`E_i = sum_t ((onehot(action_t) - p_t) outer x_t) / temperature`

At episode end, scalar reward is one for a first own hit and zero otherwise. Each
episode's baseline is the mean reward of the other three independent episodes.
The update is `1e-5 * mean_i((reward_i - baseline_i) * E_i)`, with the aggregate
function-RMS capped at .01 times initial head RMS. No per-trajectory normalization,
bootstrapped critic, autodiff, or backpropagation through time/vision/physics.
This is an analytic policy-gradient rule and should be described as such.

Temperature .028435120164498118 is fixed from a score-free calibration criterion:
5% average nongreedy probability on the existing 1,024 generic feature rows. That
is not a guarantee about on-policy exploration. Sampling and eligibility must use
the same capped projected features; greedy equivalence under radial scaling does
not imply equivalent stochastic probabilities. Action holding was considered and
rejected after a cached-frame proxy showed substantial extra behavior changes.

Training cap: 40,000 new camera transitions / 20 minutes, maximum 400 complete
batches, minimum 20 with both reward outcomes represented across training.
Last complete endpoint only; unused partial work counts. Evaluation: 64 fresh
paired 20-second games per policy, greedy at camera cadence, 10 minutes and
128,000 transitions maximum; 30 minutes overall. No outcome-based extensions.
Primary learning screen: at least +10pp first-return lift and positive paired
95% lower bound. Provisional absolute reliability: at least 80%. Rally retention
uses a separate -5pp all-return noninferiority margin. Winning is descriptive.

Sol owns implementation and focused estimator/recovery checks; Astra reviews the
prepared protocol and decisive outcomes. Record actual exploration, mixed-reward
batches, clipping, update norms and cumulative displacement. A positive result
would establish incremental learning, not superiority over an unrun matched SARSA
condition. A null cannot establish a representation ceiling, especially if
exploration or effective parameter movement is negligible.

## Research used and limitations

- [Williams (1992)](https://gwern.net/doc/reinforcement-learning/model-free/1992-williams.pdf):
  Astra read the local eligibility, baseline-independence and episodic-accumulation
  derivation. This supplies the update principle, not a Pong performance guarantee.
- [Sehnke et al. (2010)](https://archive.air.in.tum.de/Main/Publications/Sehnke2010a.pdf):
  Sol read the parameter-exploration derivation and variance discussion. Temporal
  action sampling can have high variance; the actor is a test, not an assumed win.
- [Hausknecht and Stone](https://arxiv.org/html/1507.06527v4): both reviewed the
  partial-observability motivation and comparison with observation stacking.
  Their backprop-trained recurrent architecture is not adopted here.

Detailed independent notes and diagnostics are retained at
`/workspace/m1b-experiments/pong-mechanism-review-20261022/` and
`/workspace/m1b-experiments/pong-credit-state-research-20261022/`.
The completed previous trial's journals were losslessly compressed with full
byte-reconstruction verification: 3.119GB to .425GB in 32.67s. Restore from its
archive manifest before raw-journal inspection; metadata and policies remain plain.

Status: completed; no demonstrated first-return improvement. See results below.
Exact trial: `/workspace/m1b-experiments/pong-first-hit-actor-20261022-r1/`;
source `6e556d52ef2fa3d216bd2c73f6a7c781754841e2`; original launch
1792675734506565778ns. Both outcome checks passed; source freeze released.

## Completed outcome

| Fresh paired evaluation | Carried start | Actor endpoint |
|---|---:|---:|
| First returns | 35/64 | 35/64 |
| All returns / opportunities | 51/115 (44.35%) | 61/124 (49.19%) |
| Games with at least two returns | 12/64 | 16/64 |
| Games with at least three returns | 2/64 | 5/64 |
| Games with at least four returns | 2/64 | 3/64 |

Primary first-return lift 0pp, paired 95% interval [-4.69, +4.69]. Learning and
80% absolute reliability screens fail. All-return lift +4.85pp, interval
[-1.64, +11.06]; gain is not established, but the separate -5pp retention screen
passes. One endpoint game timed out with an unfinished incoming approach. No
winning or scaffold-completion claim follows.

107 complete batches used all 39,911 training transitions, with no unused partial
batch. 428 attempts supplied 225 hits and 203 failures; 90 mixed-outcome batches
produced 90 nonzero updates. No update reached the aggregate cap. Actual sampled
nongreedy frequency was 4.788%, mean entropy .1182. Thus absent exploration,
uniform rewards and clipping do not explain this null.

Net function-RMS movement was .004975, only 1.654% of the initial head RMS.
Median nonzero step was .000530 versus the .003008 cap. The final head changed
10/1,024 greedy decisions on the generic calibration cache; mean probability
total-variation shift was .843pp. Those are cached-input measurements, not an
on-policy coverage guarantee. Net movement divided by root-sum-squared step
length was .9303, so largely canceling updates are a competing explanation to
simply too-small steps. This is descriptive, not a gradient significance test.

Execution cost: 879.43s training, 309.98s evaluation, 1189.41s total (19.82min);
33,559 evaluation transitions and 556 initial images. No worker/report failures.
Training throughput 45.38 cameras/s versus evaluation 108.26; repeated full
projection-content checks and projection arithmetic plausibly contribute, but
no isolated speed benchmark was run. Fifteen focused prelaunch tests passed.
Sol's sequential-update/source/trace check passed in 1.706s; Astra's independent
raw-count/action-probability/paired-outcome check passed in 1.567s, both with zero
new observations. Implementation integrated into main as `09b8f71`.

## Next hypothesis under continuing authority

Astra proposed conditioning the empirical baseline on the starting serve: four
ordinary episodes from the same fresh start, with independent action RNGs and
fixed weights, instead of four different starts. A leave-one-out baseline then
compares different action realizations under the same initial difficulty. All
episodes still start normally and all exposure counts; no hidden state enters
the learner. The tradeoff is fewer distinct starts and potentially many uniform,
zero-update groups. Sol is independently challenging this proposal against
update-scale, critic and representation alternatives before another allocation.
No outcome-based extension of the completed trial is allowed.
