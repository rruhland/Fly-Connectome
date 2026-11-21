# Held-out search-to-table motor transfer

Frozen search-to-table transfer won both gantry games and lost both five-axis games. Direct table-online remapping won one of four. The search branch made 57 legal returns versus 54 online, and needed more elapsed time to reach a first return in every shared success. A stronger direct control identified the entire motor map with 12 coded RGB commands and won all four games with 83 returns. [Game receipts and matched frame totals](transfer-summary.json.gz)

## Frozen protocol

Four installations crossed independent wiring, table camera, search and full-game seeds on two previously modeled arms. The two camera positions shifted together from their earlier mounts, with a newly seeded stereo baseline, focal angle and appearance. The camera calibration is public. Every branch starts from the same archived table player on its fixture. Two 34-command optical setup movements on the originally calibrated wiring supplied the shared camera reference. They did not expose the new wiring. The optics, body model and normal service and rally updates were shared. [Frozen seeds and method hashes](transfer-protocol.json.gz) · [Fixture and camera code](../../scripts/experimental/tt16_transfer_protocol.py)

The search branch ran the existing continuous hidden-source controller, starting on its original controls and stopping at the end of the first block with the new controls. Its motor correspondence came from commanded moves and the 3D RGB tool position in the last 96 frames, using the same signed candidate family and settling approximation as TT15. The frozen pitch rule then queried the proposed weakest translation port with 12 commands, including three at positive 0.70. The probe-only player received exactly that command and observation tape. It selected an axis and sign for the queried port using the TT15 one-port rule and filled the other ports from its prior. The announced probe command also reveals which port the search branch selected. The direct online branch used no new-wiring preparation and could remap from scored-point RGB. The search and one-port branches held their maps fixed in play. [Transfer and probe replay code](../../scripts/experimental/tt16_transfer_bridge.py) · [Table game code](../../scripts/experimental/tt16_transfer_game.py)

The search latency switched from zero to two frames at frame 76, then to one and three for prefixes long enough to reach those changes. Gaussian command noise started at frame 76. The dry pitch query used ordinary transport, as in TT15. Scored play interleaved ordinary, mixed, soft and delayed points with matched start seeds. The transport and the rival's service role were part of the physical fixture. Point outcomes could change who served next. Each match used the native first-person hall and alternating service to at least 11, with a two-point winning margin. [Search disturbance code](../../scripts/experimental/tt15_context_trial.py) · [Table conditions](../../scripts/experimental/tt6_continuous_trial.py)

The independent coded comparator spent 12 ordinary dry control frames moving all ports in a known, distinct code. It matched the calibrated RGB pose to signed versions of the archived model and held that choice during scored play. It received the same hall optics and the same number of dry frames as the one-port control. Simultaneous commands supplied information about more directions. The one-port pulse spent 2.1 normalized rate-frame units on the selected port. The coded pulse spent 28 on the gantry and 35 on five-axis across all ports. The search branch additionally spent 81–200 units in search. [Coded probe implementation](../../scripts/experimental/tt16_coded_probe.py)

## Whole-course accounting

Each branch pays for the shared 68 optical commands, every new-wiring dry command it uses, and every scored control frame. Search also pays for all control frames from entering its other task, including the periods before the new wiring. Time to the first legal return uses the event frame within the point and includes all previous points. A match with no legal return is censored at game completion. The unit is an elapsed 1/30-second control interval. The initial stereo images and episode resets did not advance the simulated control clock. Archived training, environment generation and CPU processing are outside the clock. The reported points per 1,000 frames divide earned points by total branch frames, including the preparation cost. The receipts preserve requested rate magnitudes per port as a separate measure of actuation effort. [Frame-accounting implementation](../../scripts/experimental/tt16_transfer_game.py)

| Branch | Wins | Points | Legal returns | Total frames | Points per 1,000 frames | Median first-return frames on successful games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Search and one-port pitch | 2/4 | 25 | 57 | 7,413 | 3.37 | 524 (3/4) |
| Identical one-port probe alone | 0/4 | 0 | 0 | 1,935 | 0 | — (0/4) |
| Direct table-online | 1/4 | 27 | 54 | 5,957 | 4.53 | 164.5 (4/4) |
| Direct coded probe | 4/4 | 45 | 83 | 7,523 | 5.98 | 127 (4/4) |

| Fixture | Search frames, all / changed | Search and pitch score / first return | One port alone | Table-online score / first return | Coded score / first return |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gantry 0 | 574 / 143 | **12–10** / 700 | 0–11 / — | 10–12 / 322 | **12–10** / 126 |
| Gantry 1 | 399 / 141 | **11–5** / 524 | 0–11 / — | 5–11 / 161 | **11–5** / 125 |
| Five-axis 0 | 285 / 148 | 2–11 / 461 | 0–11 / — | 1–11 / 168 | **11–7** / 128 |
| Five-axis 1 | 397 / 121 | 0–11 / — | 0–11 / — | **11–8** / 161 | **11–7** / 129 |

The four search branches used 1,655 frames in their earlier task, of which 553 were on the new wiring. On both gantries the search stage already found the full signed map. Its first table returns came after 46 and 45 scored frames. Coded calibration found the same maps and reproduced all 38 gantry point-action tapes. On each five-axis arm, search fit three of five axes. The frozen query targeted the wrong port on five-axis 0 and kept its starting sign. On five-axis 1 it fixed the queried sign and left two axes swapped. Coded calibration recovered all 18 signed assignments across the four fixtures. [Gantry 0 search and query](transfer-gantry-0-bridge.json.gz) · [Gantry 1](transfer-gantry-1-bridge.json.gz) · [Five-axis 0](transfer-five-0-bridge.json.gz) · [Five-axis 1](transfer-five-1-bridge.json.gz)

The two gantry wins show useful motor knowledge can carry across tasks. The longer whole-course clock and the five-axis assignment failures limit its value as table preparation on these bodies. The direct coded comparison assumes a calibrated body model and the signed-permutation family. The study changes wiring and camera geometry on two existing body designs. [Portable player and per-game evidence index](transfer-summary.json.gz)

The audit replayed all 250 scored point tapes from 16 complete games. A second acquisition reproduced the shared optical setup byte for byte and the search commands, RGB positions, scalar readings, pitch observations and decisions on all four fixtures. [Preparation replay](transfer-prep-audit.json.gz)

The gantry player after its 12–10 search-assisted win preserved all 180 reference native Pong action tapes and earned the same 41 points. Its checkpoint contains the new signed table map and the setup records the fresh camera rig for resume. [Search-assisted gantry checkpoint](transfer-gantry-0-search_pitch-player.json.gz) · [Native Pong replay](transfer-pong.json.gz)

## Reproduction

Use the [TT6 simulation environment](../tt6/README.md) with the experimental scripts on `PYTHONPATH`.

```sh
python -m scripts.experimental.tt16_transfer_report artifacts/tt16 --freeze
for fixture in gantry-0 gantry-1 five-0 five-1; do
  python -m scripts.experimental.tt16_transfer_protocol artifacts/tt16 "$fixture"
  python -m scripts.experimental.tt16_transfer_bridge artifacts/tt16 "$fixture"
  for branch in search_pitch probe_only table_online; do
    python -m scripts.experimental.tt16_transfer_game artifacts/tt16 "$fixture" "$branch"
  done
  python -m scripts.experimental.tt16_coded_trial artifacts/tt16 "$fixture"
  python -m scripts.experimental.tt16_coded_trial artifacts/tt16 "$fixture" --game
done
python -m scripts.experimental.tt16_transfer_report artifacts/tt16
```
