# Iterative population search in the optical receiving twin

Iterative search synthesized **159 distinct nominally legal command tapes versus 112** under equal-budget static sampling across 13 matched physical forks. Both policies saw the same 45 legal tapes in their first batches. The added proposals remained concentrated in the same four cases and yielded **no physical same-offer rescue**. The one iterative replacement selected for a real fork missed without touching the ball. The routine scored control remained scored. [Sealed fork roster](protocol.json) · [Real issued fork tapes](forks.json.gz) · [Model and referee comparisons](report.json)

On the four supplemental normal starts selected from untreated playability records, the iterative controller made **25 returns versus 19**. All of the increase came from one single-launch point, 6487, which rose from **5 to 11**. At the first differing command, tick 357, both versions received the same far contact and near bounce for leg 3. Both scored it, but the iterative controller changed the near contact and outgoing ball. Its first 36 nominal candidates had no legal options at that decision. The next two batches found two distinct legal options, and the robust issued path came from batch two. This is a verified longer rally reached after a different outgoing strike, not a return of an incoming shot stock missed. The other issued replacement, on 10339, created a touch but did not score the shared incoming delivery. [Playable roster](playable.protocol.json) · [Complete paired point tapes](playable.games.json.gz)

Both action-divergent pairs independently replayed the native referee events. Together with the original viable control replay, all 19,667 CAD samples were positive with a minimum clearance of 15.25 mm. Across the four viable starts, nine extra search windows used 92.3 CPU seconds for complete iterative play versus 67.3 for ordinary play. The controller kept 46 ordinary supported decisions without launching a population search.

The original six-game schedule gave only one measured launch that produced an opponent delivery. Both controllers returned five in that point and zero on the other five starts. The supplemental four-game schedule was frozen separately because the first roster had little eligible play. It used prior untreated records, no TT83-treated outcome, and fresh broad opponent draws. Across both schedules all 20 points reached native termination. [Original full starts](games.json.gz)

## Search and controls

The action modes come from joint/time correlations in 34 actually paid broad movement paths and the 18 observed differences between physical failures and scored alternatives. Add explicit smooth per-axis exploration directions. The common first batch uses stock candidate roots, encoded transport from the sensitivity bank, and camera-/encoder-driven body forecasts. Both methods use 36, 36, and 32 nominal trials with the same final nine-path screen and 18 perturbation trials. Distinct tapes are counted on the compiled command sequence at six-decimal resolution. Rechecking the same elite does not count as novel.

The iterative variant retains diverse near-miss and separated-contact elites. It steers some descendants toward the blade-plane gap and estimates a shrunken covariance in the learned joint/time basis. Paths remain in a bounded trust region and obey the normal actuator slew and range. Up to four survivors warm a following search. The equal-budget control spends its later trials on independently spread proposals about the original roots. An ordinary move with an uncertainty vote retains priority without an extra population run. A population action must be nominally legal and legal under at least one public-model error setting. The body and flight estimate enter from encoders and cameras. Actual hidden state is available only to the referee and independent matching audit.

## Exposure rate and dense optical estimate

The higher-rate TT75 observer was paired with the same search on four fixed fork prefixes. Its causal flight sampling made 675 usable stereo observations in 699 exposures, but a coherent fresh fit was available at just one of four decision ticks. The other three used the original optical track. On 8544, the fitted dense packet kept the optimizer from issuing the known legacy false positive. No paired score changed. This is evidence about exposure-to-fit coverage and an isolated false accept, not a demonstration of improved full games with dense optics. [Dense source protocol](dense.protocol.json) · [Physical paired assessment](dense.forks.json.gz) · [Exposure receipt](dense.sensor.log)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt83_population_eval plan --feeds /tmp/tt25-rallies.json
python -m scripts.experimental.tt83_population_eval forks
python -m scripts.experimental.tt83_population_eval games
python -m scripts.experimental.tt83_playable_course plan --feeds /tmp/tt25-rallies.json
python -m scripts.experimental.tt83_playable_course run
python -m scripts.experimental.tt83_dense_combination plan
python -m scripts.experimental.tt83_dense_combination forks > artifacts/tt83_population_search/dense.sensor.log
python -m scripts.experimental.tt83_population_report --verify
python -m pytest -q tests/test_tt83_population_search.py
```
