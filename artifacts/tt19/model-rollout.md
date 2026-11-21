# SO-101 receiver with an independent factory model

The receiver uses calibrated RGB position and velocity to initialize its own simulation of a ball and a factory-neutral SO-101 arm. It shortlists up to 32 commands from a catalogue of 2,048 dry arm movements and runs the shortlisted motions with MuJoCo contact, table restitution, drag and Magnus force. The six best nominal candidates get two additional simulations with perturbed ball position, velocity and spin. The receiver prefers strokes that cross the net and bounce legally on the opponent's half. The separate simulator disables opponent racket contact. [Receiver](../../scripts/experimental/tt19_model_rollout.py)

The player reads ball tracks from the native wrist and fixed RGB cameras. The receiver has the public station model, the mounted paddle calibration and the measured dry-model atlas. It starts each forecast at neutral, estimates spin from optical curvature when the track is stable, and uses a bounded spin prior otherwise. It receives no live simulator state. The serving branch reuses the local SO-101 stroke bank learned from RGB and point reward. The matched nominal condition uses the same atlas and serving bank with the nominal flight and racket prior. The gantry condition uses the previously learned flight law and racket impulse prior. [Checkpoint](match-model-rollout-native-player.json.gz) · [Atlas](factory-dry-catalog.npz)

The serving branch drew on 500 local search points and 504 repeated qualifying points totaling 38,526 control frames. The collection also recorded the receiving-side practice for the shared bank, bringing total local collection to 2,008 points and 82,341 frames. The final receiver used its separate simulation for 2,048 dry movements of 20 frames each. The trial charged that separately and performed no additional target practice before the held games. [Search receipt](context-local-practice.json.gz) · [Consolidated bank](consolidated-local-bank.json.gz) · [Dry catalogue build](catalog-build.json.gz)

| Held game seed | Rollout score | Legal returns | Legal own serves | Analytic score | Analytic returns |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 424520913 | **11–5** | 6 | 7 | 7–11 | 5 |
| 56883013 | 6–11 | 6 | 8 | 3–11 | 2 |

These are strict full-spread games with the same first server and learned serving bank. The analytic comparison used the same factory arm atlas and source priors and ranked candidate trajectories with an analytic ball-contact approximation. Both controllers used native RGB. The analytic controller had one point with two legal returns. The rollout receiver contacted all 17 receiving points across its first two games. The median error in its predicted first contact time was 0.125 control frames. The real player returned 12 of the 15 modeled legal shots. Loading the archived checkpoint reproduced the commanded actions and plans from a saved optical stream in each game. [Initial rollout games](match-model-rollout-native.json.gz) · [Matched analytic games](match-model-rollout-analytic-native.json.gz) · [Checkpoint replay](model-rollout-initial-checkpoint-replay.json.gz)

| Extra held seed | Gantry prior score | Nominal score | Legal returns, gantry / nominal | Multiple-return points, gantry / nominal |
| ---: | ---: | ---: | ---: | ---: |
| 61838512 | 9–11 | 9–11 | 7 / 7 | 1 / 1 |
| 42786193 | 7–11 | 7–11 | 9 / 9 | 1 / 1 |
| 1293821 | **13–11** | 6–11 | 12 / 8 | 2 / 1 |
| 4472301 | 9–11 | 7–11 | 6 / 5 | 1 / 0 |

Across the six held games, the gantry-prior receiver won twice, landed 46 legal returns and played five points with multiple legal returns. In the four matched games, it won one game to the nominal control's zero and made 34 versus 29 legal returns. At seed 1293821 it won both receiving points with two legal returns and the whole game after deuce. The matched sample provides limited evidence about transfer reliability. [Complete qualification receipt](model-rollout-qualification.json.gz) · [Gantry-prior extra games](match-model-rollout-extra-native.json.gz) · [Nominal extra games](match-model-rollout-nominal-native.json.gz)

The gantry-prior controller made contact on all 59 planned receiving points. Its median first-contact timing error was one MuJoCo substep, or 0.0625 control frames. It made an actual legal return on 41 of the 51 receiving points where the model predicted a legal one. It used 5,030 internal simulations across the six games. All 224 point tapes across the gantry, nominal and analytic games reproduced their physical scores and events. The compressed dry atlas reproduced byte for byte, and the saved rollout checkpoints reproduced the action plans from stored optical measurements on selected points in all ten rollout games. [Qualification and replay](model-rollout-qualification.json.gz)

An isolated replay of four RGB prefixes took 0.63–0.73 s per stroke decision on one CPU, including the candidate search and the two-stage physical simulation. Game time in the receipts uses the 30 Hz control clock. The model rollouts ran slower than real time.

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt19_model_rollout_trial artifacts/tt19 \
  artifacts/tt19/factory-dry-catalog.npz --seeds 424520913 56883013
```

Use `load_checkpoint` with the compressed player file and atlas to reconstruct the serving bank, priors and receiver. The loader checks the atlas digest and factory model revision. Run `python -m scripts.experimental.tt19_model_rollout_report artifacts/tt19 --rebuild-atlas` to rebuild the dry atlas and recheck the complete archived games. [Trial and loader](../../scripts/experimental/tt19_model_rollout_trial.py) · [Independent atlas replay](model-rollout-atlas-replay.json.gz)
