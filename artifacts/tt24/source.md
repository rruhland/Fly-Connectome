# Pong visual duration skill

The portable checkpoint estimates whether a near-term optical approach will be followed by continued play. Native Pong supplied RGB, anonymous commands and elapsed point duration. Randomized command aliases, image orientation and palettes changed independently by launch. A separate native optical system identification step associated the controllable component with the issued commands and measured its response. The portable reader takes measured trajectories and apparent surface extent as inputs.

The reader describes relative position and recent motion in units of the controlled surface's visible half-span and the other object's visible speed. A regularized linear fit over a fixed smooth basis learns from these measurements. A practice cue counts as continued if its point runs at least 0.54 seconds past estimated travel to the controlled object. This definition uses the clock and point termination. Native hit and contact data enter the referee summaries after play. The cold reader supplies a constant. The shuffled reader uses the same cues and an elapsed-time and approach-range-stratified permutation of the duration labels. Both learned readers use the same native support gate.

| Frozen source evidence | Pong | Cold | Shuffled |
| --- | ---: | ---: | ---: |
| AUC on 477 untouched optical cues from 120 native launches | 0.850 | 0.500 | 0.684 |
| Brier loss on those cues | 0.158 | 0.249 | 0.230 |
| Mean seconds over 144 new paired native points | 4.500 | 2.964 | 3.718 |
| First returns over 141 near-side opportunities | 73 | 34 | 54 |
| Later returns | 50/81 | 10/35 | 27/60 |
| Earned points after a learner return | 0 | 0 | 0 |

The native training course had 576 launches and 2,471 qualifying cues. The native controller used the same independently fitted optical motor-response model in each arm. Its short-horizon forecasts changed only the visual geometry expected under each anonymous action, which each reader scored by its own frozen weights. A separate 54-launch validation course selected the 7-frame forecast horizon. On held launches the paired duration increase was 1.536 seconds versus cold, with a 95% paired bootstrap interval of 1.172 to 1.910, and 0.782 seconds versus the shuffled readout, with an interval of 0.470 to 1.108.

A separate 48-launch native course ran twice with the frozen source, before and after downstream wrist and table practice had used the reader. Every action digest, optical-track digest, duration, and native referee record matched across the two runs for each treatment. Retained mean durations were 3.498 seconds with the Pong reader, 2.293 seconds with the cold reader, and 3.061 seconds with the shuffled reader. This establishes behavioral retention of the separately stored readout under read-only downstream use.

In a separate descriptive wrist audit, five continuous joint readouts ran on each of six fresh physical starts. The Pong score from four-frame wrist RGB windows through control tick 8 ranked 25 of 26 comparable differing-outcome pairs correctly. The shuffled score ranked 19 correctly and cold gave ties. There was native support on 77.2% of those 123 optical windows. The top Pong score chose the unperturbed candidate at all six starts, limiting the audit to evidence about predictive ranking.

The wrist and table rally assays feed their own optical measurements to the same frozen reader. The reader's soft support envelope discounts unusual values coordinate by coordinate. Joint distribution shifts need paired target validation.

## Table-tennis credit

All treatments used the same 25-command teacher opening and the same demonstration-assisted RGB and joint-feedback pupil. That pupil had twelve demonstration tapes across five launches, with 1,699 post-handoff samples. The table experiment used fixed stereo RGB at 240 by 180 pixels per eye. The target practice physically evaluated seven bounded continuous joint-rate perturbations on each of three separate launches. The public outcome consisted of elapsed play and the point result. The extra source credit came from four-frame measured optical pairs from the two cameras, using the higher-supported view for each cue. The reader scored eight control ticks after first support. True and shuffled scores changed the practice fitness assigned to the very same camera and motor records.

All three treatments selected the same feedback readout. The pupil reached its second legal return on two of six archived held points and two of three physically fresh eligible points. Its longest alternating sequence was six on the fresh course. All 27 treatment tapes passed native command replay verification. Identical learned readouts shared physical execution. This matched assay found a null effect of the added source credit on joint-policy selection.

In an exploratory follow-up, a local regularized fit used the three signed pairs of practice perturbations to update the continuous joint feedback from the same public-reward anchor. It used the same step scale and practice budget across treatments. All three resulting readouts produced distinct live post-handoff commands. On four new eligible starts, every readout reached a second own return. The Pong, cold and shuffled treatments made nine, ten and eleven own legal returns, with longest alternating sequences of six, six and eight. All twelve independently executed tapes passed native replay. The Pong credit produced motor changes without improving this table-tennis endpoint.

The next test can focus credit on a directly observed consequence of an intervention. Apply matched short signed pulses at the same native optical state, wait for camera evidence of each pulse's actuator effect, and learn which subsequent changes in the other object's recurrence predict continued play across the branches. The target body can recover its own pulse effect and delay from its cameras. This tests learned timing and credit assignment near the action that mattered, which the episode-level feedback comparison averages away.

## Reproduction

Run these modules from the repository root in the table-tennis environment.

```sh
PYTHONPATH=src python -m scripts.experimental.tt24_source_native qualify
PYTHONPATH=src python -m scripts.experimental.tt24_source_native retention-pre
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt24_source_bounce_audit
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m scripts.experimental.tt24_source_rally
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m scripts.experimental.tt24_source_rally_continuous
PYTHONPATH=src python -m scripts.experimental.tt24_source_native retention-post \
  --evidence artifacts/tt24/source-wrist-audit.json.gz artifacts/tt24/source-rally-credit.json.gz \
  artifacts/tt24/source-rally-continuous.json.gz
```

The portable weights are in `source-weights.json`. The practice, held native, frozen revisit, and wrist audit receipts are in `source-practice.json.gz`, `source-held.json.gz`, `source-retention.json.gz`, and `source-wrist-audit.json.gz`. The table-tennis credit receipts are in `source-rally-credit.json.gz` and `source-rally-continuous.json.gz`.
