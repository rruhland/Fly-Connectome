# A shared motor correction for Pong-guided opening contacts

The context-conditioned correction converted 25 of 48 newly held Pong-guided serves, up from 19 with the stock stroke and 17 with a correction that pooled all contexts. The conditioned arm made contact on 45 starts versus 46 with the stock stroke. Own-half faults dropped from 19 to 11. The same learned correction converted 11 openings with neutral lateral gains and two with permuted Pong gains. [Design](opening-motor-design.md) · [Held commands and scores](opening-motor-held.json.gz)

| Signed lateral choice | Stock contacts / legal returns | Pooled contacts / legal returns | Conditioned contacts / legal returns |
|---|---:|---:|---:|
| Pong duration | 46 / 19 | 39 / 17 | 45 / 25 |
| Neutral | 16 / 11 | 19 / 11 | 20 / 11 |
| Permuted Pong | 2 / 0 | 4 / 2 | 3 / 2 |

For the Pong-guided arm, the conditioned correction added returns on 14 paired starts and lost them on eight. Its +12.5 percentage point gain had a stratified paired bootstrap 95% interval of −6.3 to +31.3. Compared with the pooled correction, it added eight returns and lost none (+16.7 percentage points, interval +6.3 to +27.1). With both policies using the same conditioned correction, the Pong preference converted 15 starts where neutral failed, and neutral converted one where Pong failed (+29.2 percentage points, interval +14.6 to +43.8). The median depth of legal Pong-guided landings increased from 30 to 41 cm beyond the net.

## Physical motor practice

The learner extracted motion from eight synchronized stationary-command stereo frames and used the signed gain to request lateral placement. It then selected among clock-stroke corrections to advance, forward position, loft and tempo. The model used the shifted keyframes that the arm could actually command and an optical approach forecast. The fit pooled all three lateral policies, using actual legal returns and landing placement as labels. It used a regularized least-squares solve over each alternative's observed reward minus that of the same seed and signed choice with the stock stroke. The selected corrections kept solved peak lateral displacement within 3.8 mm of the stock translated keyframe. [Learner](../../scripts/experimental/tt23_opening_motor.py) · [Model checkpoint](opening-motor-checkpoint.json.gz)

Forty-eight different practice launches supplied 716 physical attempts and 27,306 simulated control frames, about 15.2 minutes at 30 Hz. For each launch, the three lateral policies received the same sampled correction IDs and eight-frame RGB prefix. Physical candidates were evaluated when the kinematic solve could preserve the lateral target. The fit used 572 matched correction-minus-stock outcomes. Held evaluation included the one failed optical approach. All 337 distinct within-policy held action segments independently reproduced the expected event digest and final outcome. [Practice details](opening-motor-practice.json.gz)

## Continuous six-start pilot

The same conditioned model and unchanged signed choices opened each new point. At tick 25 the shared stereo inverse-flight controller took over and the five-axis opponent continued playing to the strict native outcome. The original Pong-guided stroke supplied a matched motor reference. All 24 points replayed exactly. [Commands and referee events](opening-sequence.json.gz) · [Continuous runner](../../scripts/experimental/tt23_opening_sequence.py)

| Opening policy | Legal openings | Legal near returns | Alternating lengths | At least 10 |
|---|---:|---:|---|---:|
| Conditioned motor, Pong duration | 5/6 | 14 | 4, 12, 8, 0, 2, 1 | 1/6 |
| Conditioned motor, neutral | 1/6 | 1 | 0, 0, 0, 0, 2, 0 | 0/6 |
| Conditioned motor, permuted Pong | 0/6 | 0 | 0, 0, 0, 0, 0, 0 | 0/6 |
| Stock motor, Pong duration | 3/6 | 11 | 4, 12, 5, 0, 0, 0 | 1/6 |

Both Pong-guided arms reached the tenth legal bounce after 303 accumulated control frames, 10.10 seconds of simulated play. The conditioned policy chose the stock motor option for that point and issued exactly the same commands. It chose a different motor for four of the six points and produced additional legal openings on two of those and a longer exchange on a third. The pilot qualified for the 12-start confirmation on its count of ten-return points versus the same corrected arm using neutral gains.

## Independent 12-start confirmation

The confirmation schedule and decision rule were recorded while the pilot was still running, after its first partial match. The fitted motor, signed gains, service lanes and continuation remained frozen throughout confirmation. The original motor also ran on the first four starts, the prespecified balanced subset. All 28 complete command tapes replayed exactly. [Confirmatory commands and outcomes](opening-sequence-confirmatory.json.gz)

| Opening policy | Physical contacts | Legal openings | Legal near returns | Points with at least two near returns | Longest rally |
|---|---:|---:|---:|---:|---:|
| Conditioned motor, Pong duration | 12/12 | 7/12 | 14 | 5/12 | 6 |
| Conditioned motor, neutral | 4/12 | 4/12 | 14 | 4/12 | 8 |
| Stock motor, Pong duration | 4/4 | 0/4 | 0 | 0/4 | 0 |

On the fixed four-start motor comparison, the conditioned Pong policy converted two openings and landed five legal near returns. Both motor policies made all four opening contacts. In the 12-start signed-gain comparison, the conditioned Pong policy reached eight additional balls and converted three additional openings. Total near returns tied, and both arms stayed below ten. The motor correction improved opening conversion on these starts while the length advantage from the six-start pilot remained confined to that pilot.

Reproduce the motor practice and opening diagnostic with `python -m scripts.experimental.tt23_opening_motor`. Then run `python -m scripts.experimental.tt23_opening_sequence` for the pilot and add `--confirmatory` for the confirmation. These commands take the repo's Python environment, `MUJOCO_GL=egl` and one thread each for OpenBLAS, OpenMP and llvmpipe.
