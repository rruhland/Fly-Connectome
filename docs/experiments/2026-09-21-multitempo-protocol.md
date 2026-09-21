# Multi-tempo local learning, frozen transfer, then adaptation

The prior E/I boundary does not transfer across tempos. User authorizes the next
meaningful learning experiments. Preserve architecture, initial weights, measured
topology/signs, area-matched20ms excitation/5ms inhibition, frame-horizon local
rule, eight neural ticks/frame and +8tick prediction. No offline gate or classifier
is inserted or counted as learning success.

1. From the same untrained cropped checkpoint, train200 continuous trials with
   dwells2,4,6 interleaved in a fixed shuffled balanced schedule (67,67,66 trials;
   seed9060). Blank intervals12-36 frames, four cycles/trial, same two positions.
   Trial index and tempo are not neural inputs. Learning/state remain continuous
   across trials, including blank intervals. Save learned weights and full traces.
2. Freeze learning and evaluate initial, prior single-tempo3-trained, and mixed
   weights at dwells2,3,4,6 using15 trials/tempo with matched blanks seed9061.
   Three-frame dwell is held out from mixed training and interpolates trained
   tempos. Each evaluated model starts with standard warmup; continuous state
   within its evaluation. Keep first-cycle exclusion and all other quiet frames.
   Assert identical targets across models and unchanged evaluation weights.
3. Start from mixed weights with standard warmup and allow100 continuous local
   learning trials at held-out dwell3, seed9062. This measures adaptation, not
   frozen transfer. Evaluate adapted weights on the same four frozen evaluation
   sequences, including retention at the previously trained tempos. Do not select
   hyperparameters or extend training based on these evaluation results.

Primary outputs are the actual signed neural forecasts: MSE, per-polarity error
and anticipation, correct-sign counts, quiet false alarms at abs(prediction)>=.1,
and fixed controlled-task pass criteria against initial/zero/persistence. Report
all four tempos separately, not pooled. Adaptation improvement is relative to
mixed weights before adaptation on identical inputs. Existing single-tempo3
weights are a reference, not an equal-data generalization control (training frames
and tempo distributions differ). No claim that this small motif passes full M1A.

The trained events/trial are balanced, not the time spent at each tempo; slower
trials contain more quiet frames. Report total frames and trials. Save each stage
and each completed evaluation so interruption leaves inspectable results.
No learning-rule/architecture change or additional sweep is included. If mixed
learning still fails, distinguish sign, timing, and adaptation/retention issues
before deciding on a new representation or goal.
