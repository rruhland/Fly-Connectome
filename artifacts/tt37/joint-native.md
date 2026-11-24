# Training with simulated successful steering alternatives

Selecting a compatible local motor example improved later play more than openings. On 32 familiar openings with fresh reply streams, the frozen augmented pupil landed 19 later returns compared with six for the RGB-only pupil, and its best rally grew from nine to 13 alternating legal strokes. It lost two first returns in the same comparison. The gain on entirely unseen releases was smaller. [Complete paired results](joint-native-results.json)

This experiment kept the [frozen RGB-only centered pupil](joint-deep.md) and added steering examples from the mentor’s short candidate rollouts. The native examples are not additional successful RGB games. They are training-only actions tested in the physical simulator from observations made during 16 saved mentor points. The three frozen banks provided 3,628 successful incoming samples and 6,862 weak or unsuccessful samples. The model fitted only the successful labels. The negative cases are archived for future contrastive training. None of the 24 held opening IDs entered either training source. [Candidate provenance](joint-native-candidate.json)

Training associated 4,410 original RGB samples with the simulated position and velocity at the same action tick, independently reproducing the full referee event stream and matching every encoder vector. A leave-one-training-release-out camera correction reduced velocity error along the table from 0.140 to 0.132 m/s and vertical velocity error from 0.223 to 0.143 m/s. The other four measured components did not meet the predeclared five-percent improvement threshold and were left unchanged. The correction was used only to map native candidates to camera space in training. [Sensor calibration](joint-native-sensor.json)

Two controllers share the same original RGB field and the same training labels. The local average fits one setpoint around the observed ball and joints, even when nearby successful examples came from incompatible actions. The coherent variant also asks which proposed setpoint is closest to its own preceding motor reference. It fits using only neighbors from that candidate branch on the current tick. The branch can change at any tick. No stroke clock, hidden current-ball state, whole-stroke lookup, simulator planning, or coach is used in deployment. Only incoming observations can activate the new fit, and the original RGB law remains unchanged during recovery. [Controller and paired analysis](../../scripts/experimental/tt37_joint_augment.py)

On the first 12 familiar calibration points, the original RGB pupil scored nine legal near returns, averaging native examples gave seven, and the coherent candidate gave nine. The corresponding later returns were three, two and three. Coherent selection won the prespecified candidate comparison by total legal returns plus later legal returns and was then frozen. [Practice selection](joint-native-choice.json)

| Held deck and controller | Legal first | Legal later | Legal near total | Best near / alternating |
| --- | ---: | ---: | ---: | ---: |
| 24 unseen measured releases, RGB only | 8 | 2 | 10 | 2 / 5 |
| 24 unseen measured releases, coherent | 7 | 4 | 11 | 3 / 7 |
| 32 independently redrawn continuations, RGB only | 20 | 6 | 26 | 4 / 9 |
| 32 independently redrawn continuations, coherent | 18 | 19 | 37 | 6 / 13 |

The first delivered trajectories matched exactly on all 54 offered openings, as did the two no-offer referee streams. The native candidates contributed on 665 novel-deck ticks and 1,323 redrawn-deck ticks, always where the RGB field also had support. On the continuation deck, later legal returns rose from two to nine on left offers and from three to eight on right offers. The later offers themselves are not paired, because changed strikes lead to changed opponent states. The two opening losses on that deck were both in the middle lane. [Paired physical outcomes](joint-native-results.json)

Replaying the largest gain in each deck reproduced all physical events and delivered placements. Novel launch 11886 improved from one own return to three. The redrawn sequence for launch 3189 improved from one own return to six. All nine legal near contacts in the audited pair used the negative paddle face. Both points ended after an unlanded rim touch. CAD found zero arm penetrations over 8,289 physics steps and a minimum clearance of 15.1 mm. [Action-only replay audit](joint-native-audit.json)

## Reproduce

Use the pinned measured launches from the [regulation reproduction instructions](../tt29/endurance.md) and the archived native candidate and physical source receipts listed in [the augmentation manifest](joint-native-candidate.json).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt37_joint_augment calibrate
python -m scripts.experimental.tt37_joint_augment prepare artifacts/tt37/joint-native-candidate.npz artifacts/tt37/native-{54,58,67}.npz
for rule in average coherent; do
    python -m scripts.experimental.tt37_joint_augment assess /tmp/tt25-rallies.json artifacts/tt37/joint-native-candidate.npz familiar "$rule" "artifacts/tt37/joint-native-familiar-$rule.json.gz" --cap 24
done
python -m scripts.experimental.tt37_joint_augment select
for deck in held_openings fresh_replies; do
    python -m scripts.experimental.tt37_joint_augment assess /tmp/tt25-rallies.json artifacts/tt37/joint-native-candidate.npz "$deck" coherent "artifacts/tt37/joint-native-$deck-coherent.json.gz"
done
python -m scripts.experimental.tt37_joint_augment compare
python -m scripts.experimental.tt37_joint_augment audit
```
