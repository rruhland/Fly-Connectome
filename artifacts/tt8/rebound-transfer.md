# Using table-tennis rebound experience to catch a ball

A player moved a one-axis cart to catch a table-tennis ball after it bounced off a plate. In the screened game the player saw the first eight RGB images at 60 Hz. The screen hid the ball before the bounce and through the last command. After the ball crossed the catch rail on its way down, the camera showed where it crossed and the game awarded one point if the cart was within 6.5 cm. The same ball and plate contact definitions came from the MuJoCo table-tennis arena. The catcher had a separate side-view camera, layout, motor, and scoring rule. [Paired first-fall replay](rebound-same-fall.webp)

The donor was a read-only player on two table-tennis arms. From stereo RGB on the receiving side of the table, it fitted the vertical arcs on either side of each visible bounce. The estimated ratio of outgoing to incoming vertical speed is dimensionless. Four lively-plate points supplied a median of 0.916 and six low-plate points supplied a median of 0.651. The catcher used the median and robust spread of each material's optical estimates. Its other inputs were metric camera calibration, a 60 Hz clock, a nominal gravity and drag model, and the publicly visible ball and cart. It used the same post-score RGB feedback in every adaptive condition. [Table-tennis action and optical receipt](../tt7/surface-read-only.json.gz)

## When the same material carries over

The evaluator paired each target material with an independently fitted source prior. It passed the prior to the catcher without a material name. The swapped condition used the other material's optical prior. Each course ran 12 fresh sessions of eight falls per material, repeated with and without the screen. The physical opening, cart start, RGB calibration, and action budget were paired across players.

| Screened catch player | Course A / 192 | Course B / 192 | First three / 144 | First fall / 48 | Mean miss |
|---|---:|---:|---:|---:|---:|
| Matching table tennis, adaptive | 145 | 155 | 105 | 36 | 4.02 cm |
| Target only, hard prior with broad support | 139 | 150 | 94 | 24 | 5.78 cm |
| Target only, broad prior | 125 | 137 | 67 | 0 | 5.91 cm |
| Matching table tennis, frozen | 136 | 141 | 103 | 36 | 4.84 cm |
| Swapped table tennis, adaptive | 52 | 54 | 4 | 0 | 14.42 cm |
| Fixed hard prior | 83 | 89 | 65 | 24 | 18.35 cm |
| Fixed ordinary-table prior | 11 | 8 | 9 | 3 | 18.17 cm |
| Fixed broad prior | 4 | 3 | 2 | 0 | 18.21 cm |

An adaptive near-elastic control assigned 60% of its opening prior near 0.94 and 40% uniformly across the candidate rebound ratios. This stronger comparison was added after the initial broad-prior comparison, using the same held courses. Relative to it the matched donor gained 2.86 percentage points across both courses, with a paired 95% bootstrap interval of 0.52 to 4.95. Across the first three falls the gain was 7.64 points, with an interval of 1.39 to 13.89. The bootstrap sampled paired sessions separately within each course. The matched donor caught 169 lively-plate and 131 low-plate balls. The hard-prior adaptive control caught 178 lively-plate and 111 low-plate balls. [Course A](rebound-catch-a.json.gz) · [Course B](rebound-catch-b.json.gz)

Every group caught all 384 falls with the screen open. The visible post-bounce arc then let the catcher update its aim before the score. The evaluator confirmed that the cart could reach every screened and open crossing within its rate and rail limits.

## When material changes without a cue

One player per session started with an equal mixture of the two learned distributions, with no plate assignment. The plate changed after six falls, in either direction, without a visible cue or an announcement. Both materials gave exactly the same first eight ball images for a matched opening. The player updated its mixture from the terminal RGB after each fall. Before each subsequent fall, it restored 17% of the opening prior to allow for change. The three control players used the same update with hand-set priors. The generic mixture gave equal weight to ratios of 0.94 and 0.65, with spreads of 0.055 and 0.070. The broad prior used 0.75 with a spread of 0.15.

| Blind switching player | Course A / 192 | Course B / 192 | First two following the missed switch fall / 64 |
|---|---:|---:|---:|
| Mixture from table-tennis experience | 124 | 127 | 44 |
| Generic hard and soft mixture | 138 | 135 | 50 |
| Generic hard prior with broad support | 126 | 126 | 42 |
| Generic broad prior | 119 | 126 | 42 |

Every player missed the first fall after each change. The table-tennis mixture caught 22 fewer balls than the generic hard and soft mixture, a difference of −5.73 percentage points with a paired 95% interval from −8.07 to −3.65. Reuse improved screened catches when the prior was paired with the same material. On the blind switches, the best results came from the generic two-material prior and target RGB feedback. [Switch course A](rebound-switch-a.json.gz) · [Switch course B](rebound-switch-b.json.gz)

## Reproduce

Run from the repository root in the project Python environment. The replay checks the physical scoring instant, ball crossing location, each cart command, and the final miss distance. All 7,680 action tapes reproduced their recorded scores.

```sh
export PYTHONPATH=.:src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt8_transfer_trial /tmp/rebound-catch-a.json.gz
python -m scripts.experimental.tt8_transfer_trial /tmp/rebound-catch-b.json.gz --seed 72095801
python -m scripts.experimental.tt8_transfer_switch /tmp/rebound-switch-a.json.gz
python -m scripts.experimental.tt8_transfer_switch /tmp/rebound-switch-b.json.gz --seed 96039722
python -m scripts.experimental.tt8_transfer_trial /tmp/rebound-catch-a.json.gz --verify
python -m scripts.experimental.tt8_transfer_trial /tmp/rebound-catch-b.json.gz --verify
python -m scripts.experimental.tt8_transfer_switch /tmp/rebound-switch-a.json.gz --verify
python -m scripts.experimental.tt8_transfer_switch /tmp/rebound-switch-b.json.gz --verify
python -m scripts.experimental.tt8_transfer_replay /tmp/rebound-catch-a.json.gz /tmp/catch.webp 807414228
```
