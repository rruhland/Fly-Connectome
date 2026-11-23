# Reward-trained wrist policy on a six-hinge arm with a pitch jam

With the original pitch axis jammed, the carried policy and archived map averaged **17.88 consecutive rebounds**. It averaged **8.75** when its commands were padded with a zero for the extra motor, and **5.00** when both pitch axes were locked. The mapped controller reached twenty on five of eight fresh drops and moved the sixth hinge while the jammed hinge stayed at its reset angle. [Paired physical results](carried-six.json.gz)

This test carries the frozen 5×20 feedback head learned by physical rebound reward on three donor bodies. It reuses the older six-joint optical action-effect maps on the installed six-hinge MuJoCo arm. The added distal hinge has its own motor, encoder, oblique axis and inertia. The pitch jam is a joint equality constraint at the original hinge's seeded reset angle. The double-jam condition constrains the distal hinge as well. [Reward-trained head](tt24_donor.md) · [Six-hinge and jam fixtures](redundant-jam.md)

The reused maps came from two fixed camera views of painted paddle landmarks, plus encoders and self-issued commands. The old stock reference used ten signed probe blocks, or 150 frames. The archived intact, pitch-jammed, and double-jammed bodies each used twelve blocks, or 180 frames. Ball contacts were disabled during those probes. The calibration assumes the indices and starting angles of the five legacy encoders. A virtual encoder map adds the measured contribution of the distal hinge to the original five coordinates. A separate measured action-effect map translates requested virtual rates into physical rates. The benchmark supplies the matching archived map for each fixture before its first drop. This experiment charges those earlier optical maps to the transfer and adds zero probes or reward practice. [Archived stock calibration](redundant-game.json.gz) · [Archived fault-specific calibrations](redundant-jam.json.gz)

All held controllers use native wrist RGB and encoder input through the frozen stock observer, which was previously calibrated from eight generic stock probe drops with stereo RGB. The added joint contributes to the observer through the archived virtual encoder map. The landmark cameras and stock stereo pair supply no input during play. The zero-padded controls read the five legacy encoders and send the sixth motor a zero rate. Stock retention executes the unchanged five-axis controller directly. The experiment uses eight fresh matched drops with a seven-second cap, the original collision dynamics, and the original qualifying-rebound judge. Every command sequence is replayed and checked against its complete judge report. [Stock observer calibration](public-body.md) · [Pinned rosters and maps](carried-six-plan.json) · [Runner](../../scripts/experimental/tt24_carried_six.py)

## Held results

| Body and mapping | Mean longest | Total qualified rebounds | Runs of at least 10 | Runs of at least 20 | Median sixth excursion, rad |
|---|---:|---:|---:|---:|---:|
| Original stock, direct policy | 18.25 | 153 | 7/8 | 4/8 | — |
| Intact six-hinge, zero-padded | 11.25 | 118 | 3/8 | 2/8 | 0.00202 |
| Intact six-hinge, archived learned map | 14.75 | 122 | 5/8 | 5/8 | 0.02975 |
| Original pitch jammed, zero-padded | 8.75 | 88 | 3/8 | 1/8 | 0.00200 |
| Original pitch jammed, archived jam map | 17.88 | 143 | 6/8 | 5/8 | 0.03382 |
| Both pitch hinges jammed, archived double-jam map | 5.00 | 51 | 2/8 | 0/8 | 0.0000243 |

The mapped jammed condition beat zero-padding on six drops, tied one, and lost one, with a paired mean gain of 9.13. It beat the double-jam condition on seven and tied one, with a mean gain of 12.88. The original pitch encoder in the mapped jam varied by at most 0.000034 rad from reset. The distal joint's measured excursion reached 0.09161 rad, and its normalized command RMS was 0.06494. The double-jam comparison uses a separately measured archived map for its locked fixture. The small paired course measures performance using those supplied calibrations and includes the common missed opening. All 48 full judge records replayed exactly. [Paired differences and measured encoder trajectories](carried-six.json.gz)

## Reproduction

Use the repository's table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_carried_six
```
