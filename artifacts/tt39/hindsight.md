# Can a missed return teach a useful joint action?

Later stereo observations produced four physically verified repairs, but the original causal estimate repaired five of the same six missed openings. The frozen hindsight bank did not improve the new course. It landed 40 legal near returns versus 43 with the unchanged search. [Complete matched results](hindsight-results.json)

Ten practice points used fresh random opponent placements and the unchanged raw camera-and-joint controller. The recorder kept stereo ball positions, the controller’s current ball estimate, encoder angles and issued references. A missed opening was eligible only if at least four later visible observations supplied a smooth inbound trajectory that could reconstruct an earlier state. Lower openings were considered first, using camera-only height for curriculum ordering. The opponent distribution was not changed. [Pinned practice and assessment cards](hindsight-plan.json) · [Causal camera and action logs](hindsight-practice.json.gz)

Each qualifying miss had at most two checkpoints. At each checkpoint the same private nominal plant and joint search ran with either the original causal camera estimate or the estimate reconstructed from later stereo exposures. Both used the originally issued reference and encoder histories to reconstruct the body. Search proposals did not count as returns. A candidate had to resume from an exact physical action, event and encoder prefix and score a legal return on the actual simulated table. A successful hindsight candidate was then independently replayed with cameras to collect short causal joint-command samples. [All nominal proposals and physical checks](hindsight-verification.json.gz)

On the eight frozen, unrehearsed assessment releases, both controllers used the same raw actor and standard elastic search. The added prior could compete only where the stock search lacked strong support. It could try short joint fragments from a verified repair and a local posture correction, and cautiously reuse a bounded estimate residual only when the current optical evidence was similar. No future observation, card identifier or live solver state supplied a candidate at assessment. Neither controller updated its bank. [Frozen converted bank](hindsight-bank.json.gz) · [Implementation](../../scripts/experimental/tt39_hindsight.py)

Six practice openings were missed and suitable for the camera fit. At least one verified repair scored on five of those six using the original camera estimate and on four using the hindsight estimate. Hindsight added no otherwise unrepairable case. The first-ranked causal proposal at the first checkpoint accounted for all five causal repairs. Those five proposals took 55–59 nominal trials each. No alternative physical action was tried to choose those winners, but the checkpoint itself was chosen retrospectively for this comparison. The four actual hindsight repairs supplied 43 camera-and-encoder joint fragments. [Verified outcomes and source frames](hindsight-verification.json.gz)

A solver-only audit after the bank was frozen covers nine matched checkpoints. Hindsight reduced longitudinal velocity mean absolute error from 0.050 to 0.034 m/s and vertical error from 0.082 to 0.057 m/s. It worsened each position component and lateral velocity. Better estimates of those two velocity components did not translate into more verified repairs. The lowest repaired ball in the bank re-scored in an action-only replay with zero arm penetrations in 1,032 physics steps and 17.11 mm minimum clearance. [Estimator and CAD audit](hindsight-audit.json)

| Eight unseen releases, 24-stroke cap | Legal first | Legal later | Legal near | Best near / alternating |
| --- | ---: | ---: | ---: | ---: |
| Frozen stock elastic search | 7 | 36 | 43 | 12 / 24 |
| Stock search plus verified hindsight bank | 7 | 33 | 40 | 12 / 24 |

All eight first trajectories matched exactly. The bank was checked 28 times and changed actions four times across three points. Release 5082 rose from three to six own returns, and release 11127 fell from 12 to six. Both versions reached the cap on release 6090 despite different actions. Every issued bank movement passed using the ordinary camera estimate. The bounded correction learned from later vision did not authorize any movement. No adjustment was made after the held results. [Unseen scores and action differences](hindsight-results.json)

The stronger follow-up is to make the first-ranked causal search available at a live optical cue. That tests whether a controller can get the repair opportunity without ever knowing a miss is coming. [Separate prospective comparison](hindsight-prospective.md)

## Reproduce

Run against the same launch prerequisites as the [TT38 release benchmark](../tt38/joint-mpc.md).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt39_hindsight plan
python -m scripts.experimental.tt39_hindsight capture
python -m scripts.experimental.tt39_hindsight teach
python -m scripts.experimental.tt39_hindsight audit
python -m scripts.experimental.tt39_hindsight fixed
python -m scripts.experimental.tt39_hindsight learned
python -m scripts.experimental.tt39_hindsight report
```
