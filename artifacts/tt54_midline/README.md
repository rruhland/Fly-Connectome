# A physically answered low ball down the middle

The ordinary learner missed four broad-random incoming balls archived at deeply projected heights below 1.00 m and within 17 mm of the center line. Their releases were outside the TT46/TT48 scheduled training and assessment rosters and were not the previous TT40 low teachers. The nearest-camera decision state and an earlier state after the same actual far strike were replayed exactly. A finite privileged motor probe, spanning signed paddle faces and contact depths, found no legal proposals from those four ordinary bodies. On 4214 at the first logged receiving search it found a reverse-face basis and 102 contacts across 288 trials, but no landing. On 4309 it found no geometric basis at either fork. [Intact inputs](protocol.json) · [Normal-body ceiling](ceiling.json.gz)

A different posture did convert **4214, leg 4**. The unmodified, physically delivered ball was x = −8.3 mm with projected height .9818 m. Starting after the prior own-stroke separation, the teacher practiced a forward, positive-face receiving posture. It struck at actual y = −1.023 m and z = .935 m, on the rise, rather than waiting at the usual deep contact around y = −1.6 m. The opponent's original target, actual far hit and near bounce were unchanged. The return legally landed and the ordinary continuous far paddle physically answered. Independent replay sampled all 4,128 CAD states, with zero penetrations and 16.2 mm minimum clearance. An earlier deep reverse-face route did not land. Neither of the tested posture routes on 4309 landed. The extended search depth was used only by the isolated candidate evaluator, not by the plant used for this point or its audit. [Earlier pose assignment](posture-protocol.json) · [Exact actions and audit](posture.json.gz)

The prepared branch begins 1.2 s before its finish-ready state. The largest joint move was 1.879 rad. Real two-camera tracking first sees the outgoing ball cross y = 0 at tick 392, with 1.024 rad still to move on that trajectory. The first credible incoming direction is at tick 420, by which time the already prepared branch has only .335 rad left to move. The successful early preparation was privileged and used future ball information. It is not evidence that an ordinary actor could select or reach it in time on its own. [Actual RGB and encoder preparation and finish](forecourt-rgb.json.gz)

The optical packet uses the shared `RGBTaught` waveform schema. Nominal paddle geometry and encoder readouts identify the closest optically observed ball-to-blade phase. The template at virtual y = −1.10 m is extrapolated from the last uninterrupted RGB fit because the physical strike came earlier. The true contact tick and true plane coordinates are not consulted by extraction or issuance. Actual scalar legal and answered rewards authorize the stored waveform.

## What can be learned from the acquired body?

The standard RGB/encoder controller and optical retiming of the earned waveform both returned the exact teacher ball from the same physically attained posture. Both missed all four small graded neighbors and all four independent same-law TT37 draws in a fixed nearby-request rectangle. The similar contact times were not enough. The same first template could turn centimeters of incoming variation into sharply different rebounds, from strongly upward with no forward motion to softly downward despite positive forward motion. No broad block was opened. Those are matched local start-state exercises, not a whole-rally comparison. [Fixed optically phased cohort](forecourt-protocol.json) · [Paired actual results](forecourt-course.json.gz)

A bounded true-state coach ran only during practice on the four physically delivered neighboring failures. It explored signed direct drive and contact adjustments and preferred shorter radial distance among the shortlisted strokes, then submitted at most two physical candidates per neighbor. The small lateral and lower-height variants each yielded a separately landed and answered arm strike on the same original physical offer. Both scored tapes passed complete independent strict CAD verification and were then turned into two new RGB/encoder templates. Both remained rim contacts, with model-estimated radial positions of about 76 and 73 mm. Neither a center-hit solution nor robust forecourt coverage was established during practice. [Preassigned companion training and held aims](refine-protocol.json) · [Bounded physical training and RGB observations](refine.json.gz)

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt54_midline run
python -m scripts.experimental.tt54_midline_posture run
python -m scripts.experimental.tt54_midline_course capture
python -m scripts.experimental.tt54_midline_course run
python -m scripts.experimental.tt54_midline_refine run
python -m scripts.experimental.tt54_midline_transfer
```
