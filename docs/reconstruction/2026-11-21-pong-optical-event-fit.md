# Fitting a 3D racket target from optical events in Pong

A three-coefficient target fitted from visible Pong trajectory changes produced legal returns on ten of 16 receiving table-tennis points. The two matched controls used a fixed starting target or shuffled source event associations and made zero legal returns. The optical fit produced one rally with two player returns and an intervening opponent return. This was a small follow-up with two receiving courses.

## What the learner received

The source used native Pong with randomized image rotation, reflection, palette and action IDs. The fit consumed only rendered frames and the learner’s preceding actions. The shared role binder identified a track responsive to those actions and a separate track relevant to the score, using numeric comparisons previously trained from pixels, actions and native point rewards.

The event detector looked for two approximately straight visible trajectory segments whose directions reversed near the action-responsive track. It also required the inferred intersection to be near that track and the two segments to agree on where they met. The fit used 23 earlier observations from 10 candidate events in 48 random-action points. It regressed the responsive track’s eventual lateral location against a bias term, the other track’s visible relative displacement, and its projected motion. The candidate geometry assumes a controllable axis, an approximately linear approach and an optical reflection at image bounds. The event detector used geometric tolerances and trajectory continuity. It received no simulator contact or object labels.

The source fitted coefficients were approximately 0.00019, 0.886 and 0.170. The control fitted the same features after permuting the actor’s observed event positions across events. A separate native Pong course tested each candidate over the same 12 starts. The learner executed a target on all 12 starts in each condition. The public point totals were −8 for the fixed target, −2 for the optical fit and −6 for the shuffled fit. This selected the optical fit.

## Held-out outcomes

The receiving player got a single frozen numeric coefficient vector for each condition. It bound the tracks in the new scenes and learned its motor commands from the table cameras. Native contact and return records entered only the evaluator. Two fresh eight-point table courses used identical start and camera seeds across conditions. The native Pong recheck used 12 additional matched starts.

| Source geometry | Table points with contact | Table points with a legal return | Table points won | Pong opening returns | Pong later returns | Pong points earned after a return |
|---|---:|---:|---:|---:|---:|---:|
| Fixed starting target | 0/16 | 0/16 | 0 | 3/10 | 0/3 | 0 |
| Optical event fit | 12/16 | 10/16 | 2 | 4/10 | 1/3 | 2 |
| Shuffled event positions | 0/16 | 0/16 | 0 | 3/10 | 1/3 | 1 |

The receiving-table result shows that visible interactions can supply a reusable numerical target for this scaffold. The native Pong recheck shows limited Pong play from this fit and provides no evidence of retention through additional receiving-game learning.

## Reproduce

The receipt contains the role coefficients, source event windows, candidate geometry, public score selection, all paired held-out points, and SHA-256 hashes of the experiment and its shared dependencies. A second run from a copied source tree reproduced all recorded values. [Compressed optical-event receipt](../../artifacts/tt3/event-fit.json.gz)

```sh
export PYTHONPATH=.:src:scripts/experimental
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -c 'import gzip,json; print(json.dumps(json.load(gzip.open("artifacts/tt3/event-fit.json.gz", "rt"))["role_model"]))' > /tmp/event-fit-roles.json
python -m scripts.experimental.tt3_event_fit /tmp/event-fit-roles.json /tmp/event-fit.json.gz --fit 48 --score 12 --verify 12 --table 8
```
