# Does the same achieved stroke consistently leave a returnable next ball?

The first read separates the repeatable signature of an achieved stroke from the variation in the opponent's independent draw. It reuses fixed 30 Hz RGB recordings of 24 ordinary afterstates, each with six fresh target draws, and an initial 13 afterstates selected for a low original reply. Each difficult context contributes five fresh draws and excludes the original reply that selected it. Only the ball's early outbound observations enter the geometric input. The exact opponent request and any far-player images remain out of the predictor. [Initial packet snapshot](snapshot.json)

The data retain the later measured crossing at y = −1.10 m and derive height, lateral reach, forward pace and vertical velocity at the deep receiving line. The derived expected receiving failure comes from a separate ridge readout of 4,426 physically scored or missed historical deliveries from 134 release-disjoint recordings. Its release-held Brier score was .043 versus .112 for prevalence. This is a proxy for the effectiveness of an average measured receiver. It is not an actual scored arm return on these redraws and cannot account for all pose differences.

On the initial 144 ordinary redraws, the within-stroke height spread was 33.5 mm and the between-stroke spread corrected for the number of draws was 26.4 mm. The corresponding height intraclass estimate was .38 and the two independent halves of the draws agreed at .64 correlation. Estimated receiving failure was more repeatable by that split-half measure, at .77. On the initial 65 fresh difficult redraws the respective height estimates were .37 and .70, and receiving-failure split-half correlation was .69. Both within-stroke sampling variation and a repeatable stroke effect matter.

The ability to predict that effect from available evidence was weaker. On ordinary releases, fixed-size release-held models from observed outgoing arc, actual departure or accepted private candidate did not beat prevalence. In the selected difficult contexts, measured outgoing shape improved the failure proxy more than the pre-contact estimated candidate did. The difficult sample is enriched on the original outcome by design. It does not establish a critic for ordinary live play.

The additional no-render native assay selects 32 different physical source releases, excluding the entire official season, the optical snapshot releases and separately listed qualification releases. It picks an achieved, legally landed arm stroke without using the old next height. Only the earlier submitted arm commands and snapshot state are restored. The original native reply must reproduce its archived evidence. Six other draws follow the unchanged continuous response law, keeping the old original draw only for verification. This adds a less selective sample and exact native measurement without rebuilding any RGB. [Frozen native roster](native-plan.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt51_afterstate archive
python -m scripts.experimental.tt51_afterstate analyze
python -m scripts.experimental.tt51_afterstate_native plan
python -m scripts.experimental.tt51_afterstate_native run
python -m scripts.experimental.tt51_afterstate analyze --native artifacts/tt51_afterstate/native-redraws.json.gz --output artifacts/tt51_afterstate/extended-analysis.json
python -m pytest -q tests/test_tt51_afterstate.py
```
