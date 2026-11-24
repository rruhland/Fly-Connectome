# A physically earned shallow rising stroke, but no live transfer

The calibrated body does have a separate shallow mode. With a prepared left or right stance, the arm learned seven distinct early rising returns from continuous physical practice, and the opponent legally answered every one. But entering those stances during the prior outward flight did not produce a playable live controller. Across six paired measured points, the stock controller made 23 arm returns in 51 alternating exchanges. The shallow switch made zero arm returns in five exchanges. The protective high fallback did not undo the early posture change.

## Geometry and practice

The chart fits both signed paddle faces on a grid of 45 shallow sites, using broad kinematic seeds plus one static seed from the verified 14843 witness. No recorded movement sequence was used. Full CAD geometry screens the candidate poses and connects each practice standby to stock. Thirty-four sites admitted a face and 29 admitted both. For a quarter-second target-rate budget of 1.04 rad on each joint, only one of the 63 feasible signed chart solutions was within range of stock, versus eight of the right standby and ten of the left, primarily on their own positive-face wing. This is a joint-distance proxy, not a timed dynamic guarantee. The body is not just a wrist adjustment away from much of the shallow region. [Signed workspace](shallow-atlas.json.gz)

A fixed scrambled Sobol sequence sampled continuous initial position, lateral direction, speed, vertical rebound and spin. These were incoming dispenser releases, not simulated live opponent strokes. The same 24 practice recipes were independently launched from each prepared stance, in two isolated simulator workers. A privileged practice search used the known initial product of each release to propose a signed geometry and joint-position, velocity and timing correction. Every retained correction was actually executed by bounded control and only stored if native contact made a legal shallow rising return. Four right and three left strokes met that criterion and all seven were answered. Their actual contact positions ranged from −0.258 to +0.273 m across the court, from −1.244 to −1.086 m along the court, and from 0.917 to 1.010 m in height. All seven used the positive blade face. CAD replay confirmed the physical face on one example from each side and found zero penetrations in 1,560 solver states. [Right practice](shallow-practice.json.gz) · [Left practice](shallow-mirror-practice.json.gz) · [CAD witness](shallow-cad.json)

On eight untouched continuous releases, a frozen bank returned one from the right prepared stance and one from the left, on two different release IDs. Both were physically answered. Those probes still give the teacher exact practice flight and the declared initial stance. They are evidence of local motion transfer, not that the camera-only decision or the live preceding transition works. [Right probe](shallow-probe.json.gz) · [Left probe](shallow-mirror-probe.json.gz)

## Live integration and what failed

Both actors started each measured point in the stock stance. The alternative was allowed to request a connected standby from outgoing RGB flight, then use incoming RGB direction to choose a side. A private court initialized exclusively from stereo, encoders and issued references assessed the bank. The controller protected higher flights after their optical classification and retained the ordinary planner if it found no robust shallow stroke. It could not rewind body motion already issued. Three points used the unchanged continuous broad opponent box and three used a continuously drawn more shallow-biased box. In both conditions the far paddle physically reacted and drew its own target. There were five identical first physical deliveries, and the sixth point ended identically before an offer. The shallow bank was frozen and its training IDs were disjoint from this roster.

The controller tested 27 corrected motions on two live shallow arrivals, but none was nominally legal. Other candidate epochs could not find reachable geometry. It issued no learned strike. A post-run encoder replay found that on two failed low arrivals the arm was within 0.04 rad of a declared standby at the start of search. On another it was within 0.06 rad of one standby when the opponent struck, but had drifted or switched to 0.57 rad from either when shallow search first ran. The problem is not uniformly failure to reach the standby. The strong ordinary fallback also lost its first high balls once the body left stock. The actual readiness commands were replayed with full CAD and zero penetrations across 4,318 states. [Frozen bank](shallow-bank.json.gz) · [Stock games](shallow-stock.json.gz) · [Alternative games](shallow-learned.json.gz) · [Paired scores](shallow-live-summary.json) · [Achieved body and clearance](shallow-live-body.json)

The useful next test is a smaller continuous reconfiguration around the already moving stock arm, rather than making the whole game depend on choosing a shallow wing before the opponent reveals its shot. These trials did not adapt the algorithm against the held outcome.

## Reproduction

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt41_shallow plan artifacts/tt41/shallow-plan.json
python -m scripts.experimental.tt41_shallow atlas artifacts/tt41/shallow-atlas.json.gz
python -m scripts.experimental.tt41_shallow train artifacts/tt41/shallow-practice.json.gz --workers 2
python -m scripts.experimental.tt41_shallow train artifacts/tt41/shallow-mirror-practice.json.gz --workers 2 --hub 2
python -m scripts.experimental.tt41_shallow merge artifacts/tt41/shallow-bank.json.gz --sources artifacts/tt41/shallow-practice.json.gz artifacts/tt41/shallow-mirror-practice.json.gz
python -m scripts.experimental.tt41_shallow_live stock artifacts/tt41/shallow-stock.json.gz
python -m scripts.experimental.tt41_shallow_live learned artifacts/tt41/shallow-learned.json.gz --bank artifacts/tt41/shallow-bank.json.gz
```
