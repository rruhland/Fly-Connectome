# Full random games with causal joint commitment

One-shot commitment cut search CPU time by 85%, but it did not improve sustained play. Across the 24 assigned starts, the ordinary direct joint controller made 223 legal returns and the one-shot controller made 117. Adding independent, camera-credited practice raised that to 120 without improving the longest rally. None reached 100 consecutive alternating legal returns.

The comparison kept all 24 assignments from the established physical replication course, including starts without a legal opening. Each point started with one measured launch and continued under the physical referee to its natural ending or the cap of 200 alternating legal returns. The far paddle sampled its usual random target on each physical reception. Neither the launch nor the opponent’s target was an actor input. Nothing was selected from earlier performance on individual test cards. [Precommitted cards and file hashes](commitment-replication-plan.json)

All three controllers used the same body, optics, original joint instruction bank and nominal 34-candidate proposal budget. The ordinary frozen elastic controller could revise its commands every five ticks. Causal commitment waited for a fresh camera-supported incoming rise, used the raw joint law before that cue, and issued the top-ranked short joint motion for up to 21 ticks unless departure was seen. It had one retry only if the first search found nothing admissible. This compared the complete commitment policy with the ordinary controller. It did not isolate command hold time from when and how the model was initialized. The carry arm used the same commitment controller with 321 short joint fragments from the 47 legal returns on eight separate practice recordings. Camera observations assigned credit to the motion around contact. None of the three added commands from any scored test game, within or across points. [Practice receipt](../tt37/camera-time-school.json) · [Short-course causal pilot](hindsight-prospective.md) · [Full-game runner](../../scripts/experimental/tt39_commitment_replication.py)

| Frozen controller | Assigned | Openings returned | Later own returns | Total own returns | Longest own / alternating | Searches | Search CPU s | Full point CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary elastic | 24 | 21 | 202 | 223 | 31 / 63 | 994 | 349.57 | 732.25 |
| Camera-triggered commitment | 24 | 19 | 98 | 117 | 25 / 51 | 160 | 52.47 | 267.73 |
| Commitment with fixed practice carry | 24 | 19 | 101 | 120 | 16 / 33 | 172 | 54.61 | 278.01 |

All three shared exactly 23 physical first offers. Assignment 10316 produced no legal opening and the same referee stream for every controller. Commitment won four per-assignment comparisons with ordinary control, lost 16 and tied four. Fixed carry won seven, lost seven and tied ten against commitment alone. Two ordinary points, 4266 and 8627, both reached 63 alternating returns. On those two assignments, commitment returned eight each. The cheaper policy also remained cheaper after accounting for its shorter rallies. Full-point process CPU seconds per simulated control second were 1.87 for ordinary control, 1.18 for commitment and 1.20 for commitment with carry. These are workload ratios, not a real-time deployment measurement. [Matched scores, offer identities and compute](commitment-replication-results.json)

Earlier recordings of other actors on the same 24 assignments put these results in context. The planner-free joint actor had 47 total own returns and a longest rally of 11 alternating returns. The integrated actor had 134 and a longest rally of 63. Those are separate historical references, not the matched ordinary-control counterfactual. [Planner-free tape](action-baseline.json.gz) · [Integrated tape](../tt37/integrated-replication.json.gz)

The independent referee recomputed every one of the 72 scores from the recorded physical contact, separation and landing events. All agreed. A separate action-only replay reproduced every placement, contact and score in one longest point per controller. The three replays kept one launch apiece and found zero checked arm, paddle or court penetrations across 57,483 physics samples. Minimum checked clearance was 15.63 mm in the ordinary replay, 15.26 mm with commitment and 15.28 mm with commitment plus carry. The other points were not geometry-replayed. [Independent referee and detailed geometry replay](commitment-replication-audit.json)

## Reproduce

Use the usual [measured launch prerequisites](../tt38/joint-mpc.md). The entire replication roster is frozen before scoring. The `--resume` flag only finishes unopened cards in an interrupted frozen arm.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt39_commitment_replication plan
for arm in stock commit carry; do
    python -m scripts.experimental.tt39_commitment_replication "$arm" --resume
done
python -m scripts.experimental.tt39_commitment_replication audit
```
