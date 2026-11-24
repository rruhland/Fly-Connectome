# Wrist roll between solo bounces

The clamped SO-101 paddle has two flat contact faces. We started it level over the bench with the offset wrist camera facing the dropping ball. The controller used that camera's native RGB, control ticks, encoder angles and the surveyed mount geometry. It estimated the first impact by tracking the ball down a narrow vertical release corridor, then commanded a 180° wrist roll and held the new angle. The other motor targets remained fixed. The MuJoCo referee classified paddle contacts and scored rebounds reaching at least 45 mm above impact. We counted the face change only when consecutive touches on opposite faces both produced qualified rebounds. [Controller and referee instrumentation](../../scripts/experimental/tt31_flip_practice.py)

| Matched solo condition | Drops with consecutive opposite-face rebounds | Qualified rebounds |
| --- | ---: | ---: |
| Coupled torque–speed, 1.055–1.145 m release, command a half-turn | 0/8 | 8 |
| Coupled torque–speed, same release, hold one face level | 0/8 | 24 |
| Coupled torque–speed, 0.19–0.27 m release, optical half-turn trigger | 0/8 | 16 |
| Independent torque and rate limits, 1.055–1.145 m release, command a half-turn | **8/8** | 19 |

The stock wrist roll spans −157.2° to +162.8°. Its commanded rate gives a nominal 604 ms bound for a 180° target change. Under the coupled plant, the empty sweep took 1,083 ms to get within 15° of the goal and 1,148 ms to get within 5°. Under independent limits those times were 571 and 604 ms. The highest observed empty-sweep rates were 2.73 and 5.49 rad/s, respectively. Median first airborne time was 688 ms for the high coupled drops and 340 ms for the shorter ones. The high coupled attempts all touched the blade again after moving only 94–106° and produced no second qualified rebound. On the short drops the optical trigger fired in five of eight releases. The independent-limit trials demonstrated a single controlled change of face, after which the policy held the new angle. [Physical and optical tapes](flip-practice.json.gz)

On the stricter plant, same-face bouncing gives repeated contact examples with the stock wrist. Practice on both settled faces can use separate releases and finish the turn while the ball is withheld. The earlier optical studies already recorded opposite-face contacts with a settled wrist, and found that a better solo camera fit did not improve their matched rally. [Settled-face solo calibration and matched rally](../tt29/tt29_wrist_only.md)

## Command timing

Four paired bench practice drops provided a command-aligned encoder response on an immediate channel and on a channel with a withheld lag. The minimum-error alignment across candidates of zero to four control ticks identified two ticks, or 33.3 ms. These trials used the coupled plant and produced initial rebounds. Applying the learned advance to eight fresh delayed bench drops restored the zero-delay commands actually delivered to the motors and their exact trajectories. The face-change result remained zero of eight because the motors still took the same time to rotate.

The separate table comparison reused only this integer delay estimate. Both table controllers played with the same delayed and noisy command channel, the same delayed stereo images, the same 223-entry frozen movement bank, the coupled motor plant and identical opponent seeds. Each point was capped at 12 alternating legal returns. The four measured launches came from the earlier wide-opening cohort. The experiment tests transport timing generalization to those balls. [Table evaluation](../../scripts/experimental/tt31_flip_table.py)
