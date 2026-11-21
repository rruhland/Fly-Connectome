# Learning a stationary block on a fixed lane

The drill uses the end-clamped SO-101, the steerable far robot, and the strict mini-court referee. Each arm prepares for 56 control ticks through the usual motor servos. One ball is then released using the two-bounce serve generator, with a faster pace and a deeper first bounce. Every return after the serve requires a physical paddle contact followed by the correct court bounce. Each point runs continuously after release. [Drill implementation](../../scripts/experimental/tt22_block_drill.py)

## Prepared reference

Five of 432 independently screened posture and opening combinations gave a legal first block with the arm already settled. The reference's first block met the ball at 3.31 m/s forward speed and 1.14 m/s upward speed. Its requested rates were all zero and its largest joint speed before contact was under \(10^{-9}\) rad/s. The far robot physically returned the block. A second stationary near contact occurred on all four full-point reference probes, and none landed legally. [Screen](tt22_block_drill-screen.json.gz) · [Passive full-point probes](tt22_block_drill-witness.json.gz)

The prepared push adds three rate commands following an inbound event detected in two native fixed RGB streams. It later restores the starting posture. Its fixed pulse and posture were designed using the nominal arm. The coach tested candidate far swings in a separate simulation with its own prepared copy of the near arm and ranked for a strict legal answer. The live pupil receives only the fixed event rule and RGB for this demonstration. On the single development release, five of six pulse variants produced **three legal alternating returns**. The first was a stationary near block, the second a real far reply, and the third a near push. The far reply arrived at 2.76–3.47 m/s forward and 0.86–1.04 m/s upward on those points. [Native RGB and two-arm command receipts](tt22_block_drill-push.json.gz)

## Reward-only motor learning

The scratch learner starts from joint home. It explores shoulder, elbow, and wrist requests over their public motor ranges. It uses the referee's binary legal-return credit to select and retest motor postures. Camera data, contact geometry, reference poses, and the far coach's shadow state are absent from that learner. Practice ends on the first legal return or a terminal fault. The independent full-point evaluation restores the physical coach and runs until the referee ends each point. [Training and replay code](../../scripts/experimental/tt22_block_drill_trial.py)

| Evaluation | Practice trials | Practice preparation ticks | Scored practice ticks | Neutral legal first returns | Learned legal first returns | Prepared reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identical legal release each point | 108 | 6,048 | 2,347 | 0/6 | 6/6 | 6/6 |
| Variable releases during training and held evaluation | 144 | 8,064 | 2,880 | 0/12 | 0/12 | 0/12 |
| Fixed-release posture carried to fresh variable releases | 0 additional | 0 additional | 0 additional | 0/12 | 0/12 | — |

The fixed-release learner earned its first legal practice return at trial 99. Its selected posture earned legal credit on all seven visits, with eight legal practice returns in total. The six same-lane assessment points reuse the exact physical release with new point seeds. In a full point the learned posture delivered one legal block and won after the far arm missed.

The variable-release search earned three legal practice returns. On fresh variable starts the fixed-release posture touched the ball on all 12 attempts, and only one of those reaches registered a legal serve before the touch. The neutral arm let ten of the same serves land legally. The learned posture was vulnerable to early contacts under even this small release variation. All 22 complete points across the reference, push and full-point assessments reproduced their exact referee events from the two motor tapes. [Fixed release practice](tt22_block_drill-scratch-fixed.json.gz) · [Fixed release assessment](tt22_block_drill-held-fixed.json.gz) · [Variable release practice](tt22_block_drill-scratch.json.gz) · [Variable release assessment](tt22_block_drill-held.json.gz) · [Carry to fresh releases](tt22_block_drill-transfer.json.gz)
