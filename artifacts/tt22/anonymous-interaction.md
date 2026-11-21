# Anonymous interaction transfer from Pong to SO-101

A Pong-trained predictor improved held Pong trajectory-change prediction over a single-track model. The predictor failed to identify a paddle surface in both fixed SO-101 cameras. Shuffling the practice interaction timing produced the same localization counts.

## Training and test

Native Pong provided 48 training starts and 16 separate held starts. Anonymous visual tracks supplied all ordered pairs of nearby objects. The model predicted whether one track would deviate from its current trajectory over the next three observations. Its inputs described the two tracks’ separation, relative velocity and extrapolated closest approach. The single-track comparison used the moving track’s speed, age and image position. Ground-truth role names, hit events and scalar point scores were absent from the objective.

The transferred model pooled observed trajectory changes near each grayscale probe patch on four archived native SO-101 anchor videos. The score discounted motion changes explained by the single-track model. A timing control permuted the observed motion changes across eligible moments within each track and practice video. The grayscale patch bank came from independently seeded motor probes. Four disjoint held starts tested the frozen patch memories, conditional on live grayscale tracking. The archive's original two-arm commands drove these simulated points, and every practice and held replay reproduced its original RGB hash. Simulator paddle center and visibility labels entered only the held evaluator.

## Results

The held Pong Brier score was 0.1440 for the pair model and 0.1530 for the single-track comparison, over 271 pair rows that included 57 motion changes.

| Held SO-101 actor selection | Left correct | Right correct | Both correct |
|---|---:|---:|---:|
| Pong interaction memory | 0/90 | 1/90 | 0/90 |
| Timing-shuffled interaction memory | 0/90 | 1/90 | 0/90 |
| Strongest grayscale probe response | 0/90 | 0/90 | 0/90 |
| Frozen Pong command-response role | 43/90 | 0/90 | 0/90 |

The interaction memory and timing control had the same projected errors on 88 of 90 left-eye frames and all 90 right-eye frames. The interaction choice had the same projected error as the strongest grayscale probe response on 83 left-eye and 87 right-eye frames. Target practice recorded 9,499 apparent motion changes among 13,829 candidate pairs. The high background rate includes arm fragments and tracking discontinuities, so the learned association provided little evidence to distinguish an effect surface from surrounding scene motion. The earlier grayscale assay supplied a persistent paddle-center candidate in both views on 60 of the same 90 frames across its full candidate bank.

The test used four archived points. No anonymous role signal drove a live play decision. Run `python -m scripts.experimental.tt22_anonymous_interaction_trial` after the [grayscale motor-probe assay](anonymous-babbling.md). The [receipt](anonymous-interaction.json.gz) contains source weights, hashes, practice memories and held frame errors.
