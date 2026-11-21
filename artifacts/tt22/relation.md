# Anonymous visual relationships from Pong to the SO-101

Native Pong played 144 points with held random action requests, randomized action IDs, rotations, reflections and palettes. Each observation contained the rendered frame and previous command. The only training targets came from signed point scores. The first 108 points fitted the source models and the remaining 36 tested them. [Pong observation manifest](relation-pong-observations.json.gz)

The same tracker grouped quantized image regions and associated them across frames in both games. It retained up to 12 moving or persistent tracks, then evaluated every unordered pair. The descriptors measured relative distance and speed, whether separation grew or shrank, direction agreement, relative apparent sizes, and several continuous interactions. Mean, minimum and maximum pooling discard track order. Color serves local tracking and never enters the fitted feature vector. The model also receives separate single-track and recent command-magnitude summaries. Tracks carry no names, and their identities reset each point. The data provide no contact or legal-return labels to the feature learner. [Feature and loss implementation](../../scripts/experimental/tt22_relation.py)

## Held RGB forecasting

The predictor learns whether the public score will change within the next 12 frames and separately whether the point will eventually be won. Source controls shuffled score labels across Pong windows or replaced the pair summaries with those of other windows while preserving the single-track history and correct score labels. Only learned numeric pair weights enter the destination prior. One destination control starts with zero weights and another omits all pair features. All fit budgets reuse the same target observations and the same 20 held points.

Forty already-scored unmarked SO-101 points came from four existing one-camera evaluation cohorts. The evaluation rerendered the single outside camera from the physical action tapes and checked native image hashes, outcomes and every recorded referee event. Target replay scores supplied scalar labels. The stock controllers that produced the archive used their usual factory models. The learner saw the raw images, commands and scores. [SO-101 observations and source hashes](relation-tt-observations.json.gz)

| Prediction | TT fitting points | Pong prior | Shuffled score | Shuffled pairs | Scratch | Single tracks |
|---|---:|---:|---:|---:|---:|---:|
| Score within 12 frames, AUC | 4 | .752 | .737 | .736 | .739 | .736 |
| Score within 12 frames, Brier error | 4 | .172 | .174 | .174 | .174 | .186 |
| Score within 12 frames, AUC | 12 | .760 | .759 | .756 | .759 | .730 |
| Score within 12 frames, AUC | 20 | .758 | .772 | .766 | .770 | .729 |
| Win, AUC | 20 | .607 | .559 | .550 | .562 | .598 |

The held Pong comparison found a lower termination Brier error with pairs than with single tracks, .102 versus .129, while termination AUC fell from .923 to .887. Win AUC rose from .723 to .744. The largest source termination weights act on the pair minimum of projected separation change, near-future separation and relative separation rate. With four TT points, the paired mean Brier error for termination fell by .0027 versus scratch, with a point-resampled 95% interval from −.0060 to +.0007. The gain was small and uncertain. At 12–20 TT points, pair inputs themselves generally beat single tracks for termination, while the effect of the Pong initialization remained mixed. Three of the 20 held target points were wins. [Forecast metrics and learned priors](relation-forecast.json.gz)

## Live timing choice against the standard opponent

The same unmarked single-eye station played 24 new receiving points, eight for each of three shared factory timing hypotheses. The hypotheses used the existing CAD stroke generator and differed by forecasting the incoming flight zero, two or four control frames ahead. The four learners observe the same prefix of ten outside-camera RGB frames and weight target practice by similarity between anonymous visual relationships. Only the similarity weights differ. The transferred conditions derive pair-feature importance from the magnitude of the corresponding Pong coefficient. Scratch weights pair features equally. Target choice uses only the signed point outcomes of the timing hypothesis executed in practice.

At either 12 or 24 shared practice points, all four conditions selected zero forecast advance on all 16 new held starts. All had eight first legal returns and six wins. The balanced practice had two legal returns and one win, all with nominal timing. The experiment establishes a live test of the selection mechanism. Both budgets yielded identical physical actions across conditions. All 40 distinct physical tapes reproduced complete outcomes and referee traces. [Shared practice](relation-action-practice.json.gz) · [Held timing decisions and replays](relation-action.json.gz)

## Live urgency choice in a cooperative rally

The second action test uses the TT22 rally player and physical anchor rival. The player’s stock CAD stroke generator uses two fixed RGB cameras. The additional timing decision reads only the left RGB image sequence and the preceding own commands. Twelve previously played anchor points provided matched first-ten-frame prefixes and their eventual scalar scores. The target model predicts whether a score arrives in the next 132 frames. It advances the stock flight forecast one frame if estimated urgency exceeds the practice median. All four modes receive six or twelve identical prefixes and the same score labels. The cooperative rival selects physical replies using its own simulator state.

The assessment drew a separate schedule of fresh opening seeds. The first four whose image prefixes gave a different timing choice under genuine Pong and shuffled-score priors qualified for full paired points. It took 31 prefixes. Qualification used only pre-stroke pixels and the frozen models. Both timing choices then played each qualifying point in full. This estimates the consequence of differing decisions on those starts. Each condition retained the common gentle opening spread and cooperative rival. [Urgency implementation](../../scripts/experimental/tt22_relation_urgency.py)

| Shared TT practice points | Pong prior legal returns | Shuffled-score legal returns | Shuffled-pair legal returns | Scratch legal returns |
|---|---:|---:|---:|---:|
| 6 | 2 | 3 | 6 | 3 |
| 12 | 2 | 6 | 6 | 6 |

At 12 practice points, the genuine Pong learner changed the actual motor tapes and gained one return on one start. It lost one return on each of two starts and three on the remaining start. All four modes won zero points. All eight distinct two-arm tapes reproduced their complete event traces and outcomes on replay. The live tests give no evidence of practice savings. [Held urgency decisions and physical replays](relation-urgency.json.gz)

Run the forecasts and live tests with MuJoCo, the project Python environment and a C++ compiler available.

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt22_relation_trial
python -m scripts.experimental.tt22_relation_action
python -m scripts.experimental.tt22_relation_urgency
python -m scripts.experimental.tt22_relation_urgency --verify artifacts/tt22/relation-urgency.json.gz
```
