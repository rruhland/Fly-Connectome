# Native Pong scores for SO-101 lateral contact

Four archived native Pong candidates earned signed totals of +7, +5, −4 and −4 across 104 starts each. Their displacement and motion gains enter the SO-101 stroke generator at its lateral contact coordinate. The generator takes the observed blade position, adds the observed relative displacement scaled by the learned displacement gain, and adds the forecast motion scaled by the learned motion gain. The existing portable-memory convention drops the source bias because the signed 2D response axis has no defined sign in the 3D destination. The correction is limited to six centimeters from the geometric planner's original contact target. [Native source coefficients](../tt3/discovered-numeric-source.json) · [Native qualification](../tt3/discovered-source-refinement.json.gz) · [Lateral contact law](../../scripts/experimental/tt22_spatial.py)

Both cameras are unmarked and fixed on the clamp. The supplied SO-101 segmenter and table-registration code estimate ball and blade pose. The prior takes only RGB-derived lateral coordinates and a forecast from the common flight law. It uses the original yaw, pitch, stroke search, motor-rate constraints and independent command-driven CAD twin. The rival executes real physically limited strokes and the standard strict referee counts returns. The evaluator chooses the launch family before each point. The player receives neither the chosen lane nor the true launch state. [Scored trial](../../scripts/experimental/tt22_spatial_trial.py)

The scratch policy begins with six hypotheses from the original broad, signed, zero-centered Pong generator and uniform utility. Those same six hypotheses also appear in both transfer banks. The transfer banks additionally include the four qualified Pong finalists. The chronological-source bank carries the utility associated with each finalist. The score-permuted control assigns the same scores and counts two finalists away in the saved order. The three branches use the same public terminal point reward plus 0.35 per legal learner return, capped at two returns. They update once per practice point. The source count has a weight of eight equivalent observations per finalist. [Native search](../tt3/discovered-source-search.json.gz) · [Candidate and update states](spatial-score.json.gz)

The bounded-residual practice comprises eight fresh moderate lateral launches. Eight disjoint launches with larger lateral amplitudes test the frozen endpoint. Both use the varied physical rival. A fixed unit-gain rule has the same contact target as the original geometric planner and plays all eight held starts. A fixed zero lateral target plays the first four held starts. The receipt records each seed, launch family, optical readings, applied commands, terminal result and chronological learned state. [Machine receipt](spatial-score.json.gz)

## Early score curve

Through the first four matched practice starts, Pong initialization landed a first return on three points, ten legal returns altogether, and won one. Scratch landed a first return on two points and four legal returns altogether. The score-permuted condition landed a first return on two points and five returns altogether. Scratch and score-permuted won zero points. The Pong-initialized bandit selected a broad candidate on its seven-return practice point, so the early total reflects a changed exploration path as well as use of the inherited finalists. [Chronological practice tapes](spatial-score.json.gz)

| Initialization | Practice first returns | Practice legal returns | Practice wins | Wider held first returns | Wider held legal returns | Wider held wins |
|---|---:|---:|---:|---:|---:|---:|
| Pong | 6/8 | 17 | 3/8 | 5/8 | 6 | 1/8 |
| Scratch | 5/8 | 12 | 1/8 | 4/8 | 5 | 2/8 |
| Score-permuted Pong | 5/8 | 15 | 0/8 | 2/8 | 4 | 2/8 |
| Fixed unit gains | — | — | — | 1/8 | 1 | 0/8 |
| Fixed lateral center | — | — | — | 0/4 | 0 | 0/4 |

The held Pong-initialized selector chose a common signed-search candidate after practice, with motion gain −1.095. It gained one first return relative to scratch and lost one point win. The fixed unit-gain reference returned one of eight wide starts. This bounded comparison therefore measures changes to a supplied geometric target under a difficult launch distribution. All 60 condition tapes replayed with their original strict outcomes and event sequences. [Scored wider starts and selected coefficients](spatial-score.json.gz)

## Direct lateral target

The separate direct policy sends the learned coordinate to the existing inverse-kinematics and motor-limit checks as the requested contact target. If no stereo blade reading is available within four frames, its goal is lateral center. The fixed geometry control uses the original forecast coordinate at every frame. All four branches use the same optics, yaw, flight law, and physical rival. Four paired moderate practice points and four fresh broader held points are drawn from fixed generators shared across branches. Before any direct point was scored, the dimensionless launch lanes were fixed at ±.22 and ±.50 for practice and ±.59 and ±.67 for held. The original bounded comparison covers wider held lanes up to ±.96. [Direct point tapes](spatial-direct.json.gz)

| Initialization | Practice first returns | Practice legal returns | Broader held first returns | Broader held legal returns | Broader held wins |
|---|---:|---:|---:|---:|---:|
| Pong | 3/4 | 7 | 1/4 | 1 | 0/4 |
| Scratch | 2/4 | 3 | 0/4 | 0 | 0/4 |
| Score-permuted Pong | 2/4 | 4 | 2/4 | 8 | 0/4 |
| Fixed unit gains | — | — | 1/4 | 1 | 0/4 |

The geometric control's sole held return occurred on seed 1442567172. The score-permuted player landed seven learner returns and drew seven legal rival replies there. The point lasted 307 frames and ended in a loss. The native-prior and fixed geometric controllers each landed one return at that start. The permuted mode carried a variant with displacement and motion gains +.914 and +1.166. All 28 condition tapes replayed exactly. [Direct point tapes](spatial-direct.json.gz)

## Applying the updated scores back to native Pong

The same ten numeric hypotheses reentered native Pong with their original source-weighted preferences and again with the accumulated bounded SO-101 rewards and counts included. Both policies used the original native RGB role model and motor observer for six fresh four-point matches. The unmodified source preference used the Pong finalist with displacement and motion gains +1.049 and +1.138. The updated preference chose the target-selected signed-search hypothesis with gains +0.944 and −1.095. [Native return code](../../scripts/experimental/tt22_spatial_retention.py) · [Paired native Pong outcomes](spatial-score-pong-return.json.gz)

First returns fell from 13/20 to 8/20, and later returns from 23/24 to 7/12. Each preference won six points, including two earned after a return. Native losses rose from eight to 17, and signed score fell from −2 to −11. This is interference under an operationally shared preference table. [Paired native Pong outcomes](spatial-score-pong-return.json.gz)

## Weighting new outcomes by observed motion

A shared store keeps the inherited numeric prior and all eight actual bounded-practice outcomes. Each outcome carries a score, a return count, and the ratio of secondary to primary singular values in the observed blade's consecutive displacements. The ratio ignores image origin, units and axis orientation. At each target decision the signature comes from recorded pre-search ready movement. At native Pong startup it comes from raw visual motion during the existing drive identification. Log-ratio distance gives full outcome weight through 1.10 and tapers continuously to zero at 2.00. The weighting rule uses no game or apparatus identifiers. [Shared preference store](spatial-context-player.json.gz) · [Context evaluator](../../scripts/experimental/tt22_spatial_context.py)

The context audit reproduced all eight chronological arm selections and all eight held coefficients when replaying the existing observed target input and feedback. All 16 supplied a motion ratio of 0.842 from 17 displacements. The six native matches inferred a ratio at the 0.002 floor from seven to nine displacements per match. The post-arm store gave the arm outcomes zero weight in this native context and reproduced all 24 original native action tapes and outcomes. This restores native performance on these distinct observed motion signatures and leaves the target learning curve unchanged. The target-side evidence is choice equivalence on verified physical tapes. [Exact choice and native return audit](spatial-context-audit.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1
python -m scripts.experimental.tt22_spatial_trial
python -m scripts.experimental.tt22_spatial_trial --verify
python -m scripts.experimental.tt22_spatial_trial --direct
python -m scripts.experimental.tt22_spatial_trial --direct --verify
python -m scripts.experimental.tt22_spatial_retention
python -m scripts.experimental.tt22_spatial_context
```
