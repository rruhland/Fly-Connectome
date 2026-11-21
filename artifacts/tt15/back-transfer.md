# From hidden-source search back to a table-tennis game

The simulation kept the unfamiliar motor wiring when changing games. On four fresh wirings, returning directly to table tennis with the old motor map earned one point and made zero legal returns. The learner could recover most of the wiring from movements made in a different task, where it sought an invisible source and never hit a ball. Transferring that evidence into a frozen table-tennis motor map produced one win and 20 legal returns. A 12-command calibration addressing a poorly observed direction raised that to three wins and 61 legal returns. Learning directly from normal table-tennis points was a strong comparison: it also won three games and landed 61 returns. [First-person gantry game with search transfer alone](search-to-table.mp4) · [Complete game evidence](back-summary.json.gz)

## What crossed the task boundary

We used the first two discovery seeds on each arm. Each player began searching on its old wiring. At the end of the first block on unfamiliar controls we returned it to table tennis. It had spent 128–149 control frames on that wiring. The transfer code received the search player's own commands and RGB tool positions. All four search prefixes were verified against their original continuous runs. The evaluator kept the same physical wiring for the table fixture and reset the arm to its normal opening pose. All branches on a fixture shared the table checkpoint, ball seeds, noise, latency and scoring rules. Once scores diverged, the sequence of who served could diverge.

The first transfer method assigned each port using the search servo's final local estimate. The second compared the last 96 frames of observed motion with all signed assignments of the retained table geometry at several candidate delays. It used only displacement differences. The angular response, stored stroke planner, ball tracker and calibration in the table player came from its earlier table-tennis experience. Training a new stroke or inferring the table objective in the search task were outside this comparison. The port permutation family was supplied. [Source learner](../../scripts/experimental/tt15_discovery.py) · [Motion fit](../../scripts/experimental/tt15_motion_transfer.py) · [Game harness](../../scripts/experimental/tt15_back_transfer.py)

| Starting map and updates allowed | Gantry A | Gantry B | Five-axis A | Five-axis B | Legal returns, all games |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original map, frozen | 0–11 | 1–11 | 0–11 | 0–11 | 0 |
| Final local estimate from search, initially cold, frozen | 0–11 | 0–11 | 0–11 | **11–9** | 9 |
| Final local estimate from search, initially carried, frozen | 1–11 | 0–11 | 0–11 | 0–11 | 2 |
| Recent search movements fit to table geometry, frozen | **11–9** | 0–11 | 2–11 | 5–11 | 20 |
| Same motion fit, continue optical remapping in table tennis | **11–9** | **11–8** | 3–11 | **11–4** | 48 |
| Original map, continue optical remapping in table tennis | 6–11 | **11–5** | **11–8** | **11–7** | 61 |
| Same optical probe alone, original correspondence on unobserved axes | 0–11 | 0–11 | 0–11 | 1–11 | 0 |
| Search motion plus one optical pitch probe, frozen | **11–9** | 9–11 | **11–9** | **11–9** | 61 |

“Frozen” applies to the port mapping. The ordinary rally, service and timing updates continued in all games. Under both online remapping branches, the TT13 player compared commanded actions against its own stereo observations after each point. All eight online players ended with the exact wiring. With the search initialization they made two legal returns in the combined first four starts per fixture versus nine from the original map. Transferring a nearly correct map can delay recognizing its remaining error. [Online continuation](../../scripts/experimental/tt15_reentry.py) · [Paired prefix and checkpoint audit](../../scripts/experimental/tt15_transfer_report.py)

## Why the search motion was insufficient

In an evaluator-only audit, the motion fit assigned 15 of 18 port directions correctly. It assigned every port to the correct axis. All three errors had the same form: the sign of the axis that least affects tool position was reversed. The old camera model predicts approximately 5 mm of translation per unit command on this axis on gantry and 3 mm on five-axis. The other axes predict 66–118 mm. The task's 3D display measures position in 10 mm increments and cannot directly observe pitch. Thus excellent source-task performance can coexist with a consequential mistake on a table-tennis stroke.

At the transition we selected the weakest axis by the **learned** translational sensitivity, moved its proposed port, and compared the two candidate signs against the actual first-person stereo pitch measurements. This consumed 12 unscored commands and 13 stereo pairs per fixture, 0.4 simulated seconds. The query retained the right sign in one fixture and corrected it in three. The four games then yielded nine legal returns during their combined first four starts per fixture, 61 in total, and 18 sustained points. The pure table-online control also made nine early returns, 61 total and 20 sustained points. Without the search-derived correspondence, the same 12 commands and pitch observations identified the queried port but yielded zero legal returns and one point in four games. Each player had the same table checkpoint and the same query; only the search-informed player knew how the other ports moved. The five-axis checkpoint after its 11–9 win preserved all 180 reference Pong action tapes. [Native Pong](transfer-pong.json.gz)

The targeted query was designed after seeing the position-only ambiguity on these four fixtures. [Probe and control details](pitch.md) · [One winning five-axis checkpoint](pitch-five-0-player.json.gz)

## Reproduction

Use the [simulation environment](../tt6/README.md). Run the discovery suite corresponding to the chosen arm first. The example below uses gantry, index 0. Indices 0 and 1 on `gantry` and `five` are the four test installations. The stage that fits recent motion also runs its frozen table game. Reports and their checkpoints are saved separately.

```sh
python -m scripts.experimental.tt15_back_transfer artifacts/tt15 gantry 0 harvest
for mode in fixed cold carried; do python -m scripts.experimental.tt15_back_transfer artifacts/tt15 gantry 0 "$mode"; done
python -m scripts.experimental.tt15_motion_transfer artifacts/tt15 gantry 0
for mode in search original; do python -m scripts.experimental.tt15_reentry artifacts/tt15 gantry 0 "$mode"; done
python -m scripts.experimental.tt15_transfer_report artifacts/tt15
```
