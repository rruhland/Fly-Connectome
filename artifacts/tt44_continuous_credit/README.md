# Two additional same-ball rescues from new early direct motor commands

The stock SO-101 can return **9519** and **4283** with new bounded early joint feedback, even though stock, the temporal readout and short visual preview all missed those exact balls. A different direct movement also landed 4450, which was already a known same-ball rescue. This raises the physical choice ceiling on the original 12 practice offers from five to seven. None of the four moderate low misses became legal.

## What was allowed to change

The eight historical misses in the paired-credit plan were preassigned as the training set, excluding the current TT44 fresh roster. The branch restores the full native state immediately after original far-paddle separation. The original prefix and each resumed outcome retain the same actual far impact, near bounce and randomized aim. All branches use the original SO-101 clamp, paddle, absolute and slew limits, 30 Hz stereo and unchanged 28-candidate late joint MPC.

The six motor parameters control a small forward or backward shift of the optical feature estimate and five signed early joint-reference corrections. A causal pulse switches them on when the ball moves toward the arm and fades out by y = −0.53 m. Measured joint position and previous issued reference attenuate a correction when the motor is already lagging in its direction. This changes actual early joint movement without injecting a new nominal joint posture, requesting true ball state, or widening the later search. Reported “closest gap” and departure effects are training evaluator measurements, not actor inputs.

Four spread-out initial proposals and at most two local perturbations per ball gave 38 actual native action trajectories. Each new proposal ran from the same copied state and took fresh RGB. Only delivered, physical legal returns earn positive credit. If the first screen fails, measured contact or a sufficiently close physical approach can earn limited refinements. The result does not claim impossibility outside this small movement family. [Exact-state training receipts](training.json.gz) · [Underlying paired plan and controls](../tt44_paired_credit/plan.json)

| Ordinary missed offer | All three fixed teachers | New continuous early control |
|---|---:|---:|
| 9519 | Missed | Landed |
| 4450 | Two landed | Landed with another control |
| 9080 | Missed | Missed |
| 4283 | Missed | Landed |

Independent replay from the original launches verified both stock and new event tapes and found zero full-CAD penetrations in 7,013 total solver states. The 9519 and 4283 contact alignments changed from −0.868 to −0.996 and from −0.941 to −0.990, respectively. Both actual legal rescues therefore improved alignment on the ordinary deep negative face, not by finding an alternate stance. [9519 strict replay](audit-9519.json) · [4283 strict replay](audit-4283.json)

Beginning after a visible outbound cue before the far strike preserved the matched 14502 ball. The actuator gained earlier motion, but none of the five changed or one control trajectories touched. The no-contact low outcome is not solely an artifact of beginning just after far separation. [Earlier causal test](outbound-14502.json.gz)

On 4239 a second targeted variant held the early correction through the first observed rebound. One of its six physically executed variants contacted the unchanged moderate low ball, but none landed. [Longer correction fork](rebound-4239.json.gz)

The local score-driven selector freezes at the first eligible incoming RGB and encoder row, before the corrected action begins. Failed moves dilute nearby support, and each physically distinct training offer votes once. The four ordinary controls were all retained, three by choosing identity and one by a fresh nonidentity physical replay from the same incoming state. [Frozen local model](policy.json) · [Easy controls](easy.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt44_continuous_credit
python -m scripts.experimental.tt44_continuous_credit
python -m scripts.experimental.tt44_continuous_credit --ids 14502 --outbound --refine 1 --output "$out/outbound-14502.json.gz"
python -m scripts.experimental.tt44_continuous_credit --ids 4239 --keep-to=-.91 --refine 2 --output "$out/rebound-4239.json.gz"
python -m scripts.experimental.tt44_continuous_credit_audit --ids 9519 --output "$out/audit-9519.json"
python -m scripts.experimental.tt44_continuous_credit_audit --ids 4283 --output "$out/audit-4283.json"
python -m scripts.experimental.tt44_continuous_credit_fit fit
python -m scripts.experimental.tt44_continuous_credit_fit easy
python -m scripts.experimental.tt44_continuous_credit_fit held
python -m pytest -q tests/test_tt44_continuous_credit.py tests/test_tt44_paired_credit.py
```
