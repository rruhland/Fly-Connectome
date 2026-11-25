# Causal wing preparation from a normal physical receive history

The saved encoder bridge starts from two experimentally paid paths. Live motion is not started at a recorded time. It begins after an actually observed outward played ball and only while both the outward path and the body still support an entry to the bridge. The first part is capped. A fixed stereo far-paddle readout can raise or lower how far the arm advances. The incoming optical track can then authorize deeper progress or send the arm back toward ordinary control. Every issued reference uses the installed rate cap and a local independent CAD check. Successful motion options are still subject to the existing late stock window. No learned strike has an independent early lease. [Rewarded encoder bank](../tt51_low_reverse/observed-prep-bank.json.gz) · [Stock-window low skill](../tt52_stock_repertory/low.json.gz)

## First causal test on the original 43rd offer

Both branches replayed the same normal starting game, with the original issued commands protected through completion of the previous swing. They then made fresh optical and encoder decisions. The far impact and near bounce were the original physical .953 m projected return. The prepared-stock and bridge controllers both missed. The controller made its first independent change at tick 4010. At the −.45 m outgoing gate the actual stereo readout gave low-left risk .336, projected lateral −.139 m and height .997 m. A low-confidence contradictory prediction at the next y = 0 gate reduced the path target from .85 to .327. The arm was still about .76 rad short of the demonstrated elbow position at the stock search handoff, despite the later measured inward track correctly showing a low left ball. Three late option windows ran and none could produce a legal movement from that body. Both tails were physically clear. [Native paired result and detailed public-body trace](exact.json.gz)

## Retaining earlier supported stereo through ambiguity

The second controller retained progress for a bounded time when a newer view was ambiguous, but continued to allow clear rightward or high evidence to veto further travel. During brief missing RGB fits it held the last safe target rather than starting to undo its motion. From the same natural setup, with no imposed preparation after the preceding stock swing, it made the original .952849 m ball physically legal and the unchanged opponent answered. At the last issued preparation the body was within .074 rad of the demonstrated endpoint. The frozen template was accepted at the regular stock window on tick 4078 after both perturbation checks, and actual contact was at y = −1.271 m. The clearance audit found zero penetrations in all 1,376 altered-tail samples with a 17.596 mm minimum. This is a causal paired intervention with the previous 42 actual scores reproduced, not a new autonomous 43-return record from release. [Actual controls and event proof](retained-exact.json.gz)

## Earlier held neutral suite with the first controller

The first controller rescued the physically identical .964 m left-wing redraw 515671963 from an unchanged early arm. Stock and stock plus late templates missed. The controller reached +.515 rad elbow and +1.275 rad wrist by the ordinary stock window, selected the frozen low stroke, and received a legal answer. A separate fresh actor replayed every RGB-driven command and each far prediction exactly. Another neutral, 767799421, changed in the wrong direction. Both controls returned it, while the early controller missed after investing in the wrong posture. Native totals were four of five each. This supports the conditional mechanism without showing an overall rate gain. [Five matched neutral draws](../tt51_low_reverse/live-wing-neutral-report.json) · [Independent fresh-camera positive replay](../tt51_low_reverse/live-wing-positive-replay.json)

## First live controller on fresh whole points

The fixed first design scored **12 rather than 33** across all four preassigned starts against the same broad opponent. Per-game results were 15→2, 7→7, 11→3 and 0→0. The two ties had identical commands. In both shortened points, both variants returned the same first two physical offers. An early reserve under weak evidence changed the second outgoing stroke and all later play was then reactive to that genuinely different stroke. Neither affected game selected a low template. The low finishing skill is not enough to justify speculative changes to ordinary readiness. [Complete points from release](whole.json.gz) · [Locked starts](whole-plan.json)

## Motion waits for evidence in the checked scout

The checked scout reserves a compatible path without moving its arm. In this separate candidate, only a strong supported far update by outgoing gate 0 can authorize travel. It can rejoin the recorded path faster within the same installed bounds and uses multiple compatible incoming observations before acting on a moderate high or opposite limit. Its calibration is kept separate from the two earlier actors and their reported tests.

On the same physical .952849 m ball, every command agreed with prepared stock until the first high far-paddle vote at tick 4022. This is 12 ticks later than the speculative versions. With no imposed preparation, it still reached the learned blade workspace. The frozen stroke again won the ordinary search on tick 4078, made physical contact at y = −1.271 m, returned legally and was answered. The changed tail had zero penetrations in 1,376 CAD samples. This is a successfully calibrated normal-body intervention, not a complete fresh point. [Fully causal exact intervention](checked-exact.json.gz)

The checked scout scored **16 versus 19** across its separate four unchanged full games, using fresh assigned streams. Results were 7→7, 6→6, 6→3 and 0→0. In the three ties every command matched prepared stock. On 8056 a strong-looking far stereo vote of .297 authorized preparation. The first affected incoming ball was exactly the same fourth receive, and only the stock body returned it. The actual failure arose before any learned strike was selected. This rejects broad improvement under the current classifier, even after eliminating unconfirmed motion. [Four new complete points](checked-whole.json.gz) · [Their locked schedule](retained-whole-plan.json)

An independent replay reproduced the exact .953 m interaction and the complete new points. The five unique whole tapes had zero penetrations in 20,674 checks, with a minimum 15.417 mm separation. On three explicitly known diagnostic offers the checked scout went three of three vs two of three for both controls. Its two ordinary command tapes were byte-identical to prepared stock, and the low recovery first diverged at tick 4022. [Exact audit](checked-exact-audit.json) · [Whole-point audit](checked-whole-audit.json) · [Known diagnostic controls](../tt51_low_reverse/live-wing-v3-diagnostic-report.json)

## The known 63-return starting memory exposed a shared base effect

The frozen checked controller ran once from the authenticated before-game memory and normal launch on the known 63-return episode. Its recorder initially stopped after play when raw tuple-valued native events were compared to serialized lists. The authorized deterministic evidence recovery used the same model, carry, draw and seed and persisted its full action and event row before validation. All ten events before the first changed command were identical after serialization. There were **zero numerical differences across 164 fields**. The recovered controller scored **3**. The first changed command was tick 101 and the first two physical offers were identical and returned by both controllers. The wing branch authorized no motion and selected no strike. The difference came from the ordinary partial preparation shared with this experimental comparator, which was not part of the genuine TT46 parent. An independent physical replay reproduced three arm and four opponent returns, with zero penetrations in 3,123 samples. This known result is not a fresh game assessment. [Full recovered tape](checked-record.json.gz) · [Recorder structure diagnosis](checked-record-prefix.json) · [Native score audit](checked-record-audit.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt54_live_wing
python -m scripts.experimental.tt54_live_wing_v2
python -m scripts.experimental.tt54_live_wing_v3
python -m scripts.experimental.tt54_live_wing_games plan
python -m scripts.experimental.tt54_live_wing_games run
python -m pytest -q tests/test_tt54_live_wing.py
```
