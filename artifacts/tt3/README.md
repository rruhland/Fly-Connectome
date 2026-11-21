# From native Pong to an arm holding a table-tennis racket

The [experiment report](../../docs/reconstruction/2026-11-21-3d-table-tennis.md) describes which comparisons support transfer. Each compressed JSON file is readable with Python's standard `gzip` and `json` modules. The WebPs replay actual bounded joint commands. The perspective view is for spectators. Both small images on the right are the cameras the policy used.

## Run locally

From the repository root, use Python 3.11+ with NumPy, SciPy, PyTorch, and Pillow. One BLAS thread per experiment worker keeps the small regressions efficient.

```sh
export PYTHONPATH=.:src:scripts/experimental
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/experimental/tt3_replay.py artifacts/tt3/full-table-serve-rally.json.gz /tmp/table-tennis.webp
```

The default visual tracker uses SciPy. On Linux an optional compiled implementation accelerates the same four-connected-component operation.

```sh
c++ -std=c++17 -O3 -fPIC -shared scripts/experimental/tt3_components.cpp -o /tmp/tt3-components.so
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components.so
```

On ten paired simulated points the reference and compiled paths produced identical command hashes and physics outcomes on every point. The runs took 14.03 and 7.52 CPU seconds on the test machine. [Timing and hashes](pixel-timing.json)

## Reproduce the source and the receiving games

The native source uses 48 random-action camera episodes to fit action and reward associations. A separate numeric search starts without a preferred direction in its visual displacement features and scores its candidates on 1,200 native Pong points. Repeated starts within a generation are intentional for matched candidate comparisons.

```sh
python -m scripts.experimental.tt3_pong_source --output /tmp/tt3-source --raw-episodes 48 --practice-starts 60 --practice-points 3 --evaluation-starts 36 --evaluation-points 5 --ticks 1440
python scripts/experimental/tt3_numeric.py /tmp/tt3-source/pong-roles.json /tmp/tt3-numeric --generations 5 --population 12 --starts 5 --points-per-match 4 --ticks 1240 --seed 7733814
python scripts/experimental/tt3_numeric_transfer.py artifacts/tt3/pong-source.json artifacts/tt3/numeric-pong-source.json /tmp/tt3-transfer.json.gz --seed 15537921 --matches 10 --points 14 --stable --forecast --workers 2
```

The numeric and visual checkpoints contain learned coefficients, outcome counts, and rank weights. A fresh arm discovers its body map through its own camera stream. The six-relation checkpoint is a separate, more structured baseline. [Visual and categorical checkpoint](pong-source.json) · [Numeric checkpoint](numeric-pong-source.json) · [Source-search receipt](numeric-pong-training.json.gz) · [Matched 140-point transfer](numeric-qualification.json.gz)

## Other evidence

| Question | Checkpoints, matched runs, or exact replay |
|---|---|
| Can a carried player keep earning points in native Pong? | [Adaptation by observed body response](pong-retention-body-conditioned.json.gz) · [Actual post-table checkpoint](pong-post-table-body-conditioned.json) · [Transfer with and without paid stroke practice](body-conditioned-transfer.json.gz) |
| What happens if the different bodies share just one shot preference? | [14- and 48-point numeric continuations](numeric-retention.json.gz) · [Scored-stroke interference](pong-retention-stroke-forecast.json.gz) · [Motor-model interference case](pong-retention-stroke-integrated.json.gz) |
| Does varying 3D height matter? | [High/low serves and fixed-height control](table-height.json.gz) · [Body map qualification](motor-servo-paired.json.gz) |
| Can the player rally from a full legal serve? | [Six matches with a two-bounce serve](full-table-serve.json.gz) · [Winning rally](full-table-serve-rally.json.gz) |
| Which features were learned from Pong? | [Anonymous visual tracks and distractors](visual-roles.json) · [Optical fit with shuffled associations](event-fit.json.gz) · [Source-only numeric refinement](numeric-pong-source-refinement.json.gz) |
| Does past experience help choose probes or a new goal? | [Probe rhythm](exploration.json.gz) · [Probe stopping](exploration-adaptive.json.gz) · [Ordinary-score reversal](reward-score-reversal.json.gz) · [Landing-payout reversal](reward-landing-reversal.json.gz) |
| Can the player learn to return the opponent's reply? | [Identical incoming shots with different meeting positions](recovery-same-openings.json.gz) · [Learning the timing preference](recovery-learned.json.gz) · [Score-selected cushion stroke](active-stroke-score-selected.json.gz) |
| Does another rate-controlled joint immediately improve play? | [Shuffled four-actuator interface](wrist-novel-actuator.json.gz) · [Learning optical motion effects](wrist-optical-effect.json.gz) |

The earlier traveling-button Pong experiment uses separate checkpoints and a different camera and body. [Travel and 50 ms delay report](../../docs/reconstruction/2026-11-21-axial-travel-and-camera-latency.md)

## Numerical policy through a full cycle

The refined source runs the earlier numerical candidates on additional disjoint native starts. On the receiving side, utility profiles are indexed by the measured movement span. The learner also updates a secondary timing preference from visual continuation and ordinary score. The `--all-times` variant lets it propose receiving positions on both sides of a predicted bounce.

```sh
python scripts/experimental/tt3_numeric_source_refine.py artifacts/tt3/pong-source.json artifacts/tt3/numeric-pong-training.json.gz /tmp/tt3-refined
python scripts/experimental/tt3_cycle.py artifacts/tt3/pong-source.json artifacts/tt3/numeric-pong-source-refined.json /tmp/tt3-cycle.json.gz --all-times --seed 50135910 --native-seed 43029713 --matches 8 --points 20 --native --workers 2
python scripts/experimental/tt3_transfer_figure.py /tmp/tt3-cycle.json.gz /tmp/transfer.png
```

[Refined source](numeric-pong-source-refined.json) · [Full 160-point table and 240-point native comparison](refined-unrestricted-cycle.json.gz) · [Three-return real command tape](refined-sustained-rally.json.gz) · [Medium-pace opposition](partner-robustness.json.gz)

## Final experiment with learned reward attribution

The final source and receiving player use the classifier fitted to reward times. The separate fresh control learns its own classifier from its table experience. The native search can be rebuilt by giving `tt3_numeric.py` the discovered role checkpoint and the same search flags shown above, then giving its training receipt and the discovered role checkpoint to `tt3_numeric_source_refine.py`.

```sh
python scripts/experimental/tt3_cycle.py artifacts/tt3/discovered-role-model.json artifacts/tt3/discovered-numeric-source.json /tmp/discovered-cycle.json.gz --reward-clock artifacts/tt3/discovered-reward-clock.json --all-times --seed 93714422 --native-seed 81752634 --matches 8 --points 20 --conditions trained fresh_goal scratch cold_clock --native --workers 3
python scripts/experimental/tt3_table_checkpoint.py /tmp/discovered-cycle.json.gz /tmp/three-joint-player.json.gz
python scripts/experimental/tt3_long_cycle.py /tmp/discovered-cycle.json.gz /tmp/three-joint-player.json.gz /tmp/continued-table.json.gz --points 64
```

The checkpoint saves the camera associations and their supporting examples, point-score estimates, spatial coefficients, table timing, learned visual surface statistics, arm regression and random-generator states. Loading it resumes at the next point with the same physical installation. The verifier compares every requested action and every physics event in an additional point.

[Source role weights](discovered-role-model.json) · [Reward attribution model](discovered-reward-clock.json) · [Pong search](discovered-source-search.json.gz) · [Numeric score refinement](discovered-source-refinement.json.gz) · [Final source checkpoint](discovered-numeric-source.json) · [Complete matched cycle](discovered-cycle.json.gz) · [Resumable post-table player](three-joint-player.json.gz) · [Winning first-point replay](discovered-winning-rally.json.gz)

[64-point continuation from the verified saved arm and second paired native check](discovered-long-cycle.json.gz)
