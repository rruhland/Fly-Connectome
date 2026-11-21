# Two search-guided motor queries at table handoff

On two holdout wirings, two individual RGB port queries resolved the remaining errors in the search-derived motor map. The true signed key ranked second from search motion on each arm and first after the queries. A branch given the same pair of command and RGB tapes without search retained an incomplete map on each arm. Both controls and the search-assisted branch use the corresponding archived table body geometry. [Frozen holdout and rule](sparse/sparse-protocol.json.gz) · [Gantry evidence](sparse/sparse-gantry-2-evidence.json.gz) · [Five-axis evidence](sparse/sparse-five-2-evidence.json.gz)

## How a query was chosen

The learner retained the full ranked set of signed assignments from the frozen TT15 displacement fit. It converted each candidate's error into a weight using a fixed 12 mm scale. It then selected the port with greatest uncertainty over the possible axis and sign. After measuring the first port it updated the weights with centered RGB position and pitch errors, using a fixed 5 mm scale, and queried the most uncertain remaining port. Each query used the original 12-command pulse, with three positive 0.70 commands on one port. The search-free branch evaluated the exact same two observed histories, resolving unsupported assignments toward its original motor map. Evaluator wiring was available for the audit after both decisions. [Ranked model](../../scripts/experimental/tt16_sparse_model.py) · [Physical query and game protocol](../../scripts/experimental/tt16_sparse_trial.py)

The two five-axis keys had ranked ninth and second in the preceding four-fixture suite. The two queries moved both to first there, and all four development keys became exact. These were development observations. The holdout seeded a different camera rig, wiring, target course and complete scored game for each body. On both holdout search courses the reward and motor wiring changed together. Search used the earlier changing-delay and noisy transport. The unscored queries and coded comparator used the same ordinary dry transport, and all three official games on a fixture used the same disturbance and ball-seed schedule. [Earlier fixtures](transfer.md)

## Cost of the additional evidence

The two single-port queries used 24 dry frames and an absolute normalized rate sum of 4.2 across the ports. The independent coded comparator used 12 dry frames and 28 rate-frame units on gantry or 35 on five-axis. Relative to coded calibration, the two-port procedure used 85% and 88% less commanded motion at handoff. It used twice the elapsed dry frames. Counting the preceding search also adds 278 frames and 77.2 rate-frame units on gantry and 286 frames and 116.1 units on five-axis. All branches also paid for 68 shared old-wiring optical setup frames. The rate sum measures requested actuator motion. Energy and wear were outside this simulation. [Gantry costs and motion tape](sparse/sparse-gantry-2-evidence.json.gz) · [Five-axis costs and motion tape](sparse/sparse-five-2-evidence.json.gz)

## Complete scored games

| Fixture | Two ports with search score / returns / first-return frames | Same two ports alone | Coded score / returns / first-return frames |
| --- | ---: | ---: | ---: |
| Gantry, queried ports 1 then 0 | 9–11 / 22 / 419 | 0–11 / 0 / censored | 9–11 / 22 / 129 |
| Five-axis, queried ports 0 then 2 | **11–9** / 14 / 421 | 0–11 / 0 / censored | **11–9** / 14 / 123 |

Search-guided and coded calibration produced the same key and exact scored action tapes on both arms. The old-wiring setup, search and dry frames are included in first-return times. All of the elapsed difference came from preparation, with 290 extra frames on gantry and 298 on five-axis. Across both games this was 4.65 points per 1,000 frames for search-guided calibration and 5.39 for coded. The comparison shows that search can replace broad motor excitation at handoff in these two fixtures. Starting from the archived player, direct coded calibration still cost fewer frames and lower total command effort. This is one complete fixture per arm with an existing calibrated body and signed-permutation changes. The audit reconstructed the optical selections and replayed all 102 scored point tapes. [Gantry game](sparse/transfer-gantry-2-search_sparse.json.gz) · [Five-axis game](sparse/transfer-five-2-search_sparse.json.gz) · [Exact key and tape audit](sparse/sparse-summary.json.gz)

## Reproduction

Use the same environment as the [four-fixture transfer comparison](transfer.md). The frozen sparse runner writes one complete game and a portable checkpoint per branch on each arm.

```sh
for fixture in gantry-2 five-2; do
  python -m scripts.experimental.tt16_sparse_trial artifacts/tt16/sparse "$fixture" collect
  python -m scripts.experimental.tt16_sparse_trial artifacts/tt16/sparse "$fixture" qualify
done
python -m scripts.experimental.tt16_sparse_report artifacts/tt16/sparse
```
