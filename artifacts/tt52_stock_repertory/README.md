# Learned strokes inside the stock receiving window

The prior live test showed the same physical second offer on 12068 was lost by an early learned commitment at tick 158. Keeping the reversible preparation and disabling just that commitment physically recovered the ball. This test moves the same body-conditioned waveform field inside the existing stock search, without modifying the shared stock implementation, its random proposals, its cadence, or its confirmed ordinary commitments. A learned waveform can only fill a still-unresolved stock slot after an optically detected bounce and fresh stereo. It receives the same physical camera-initialized consequence tests, needs both perturbation checks and gets the ordinary lease. [Prior matched failure isolation](../tt51_partial_finish/isolated-12068.json.gz)

The field also accepts two previously paid low waveforms. Their extraction uses observed RGB motion, nominal forward kinematics from the arm's own encoders and actually issued joint references. Their replanning uses current path, timing and body support. It does not read the native target, an archived handoff marker or a true ball trajectory. New within-point ordinary waveforms are stored only after real legal credit. [RGB-only low curriculum](../tt51_low_reverse/course.json.gz)

## Exact physical preservation result

The new installed options controller reproduced every command through tick 157 on 12068 and received that same physical far impact and near bounce. It returned the second ball and the opponent legally answered. Prepared stock and stock with options each made two returns and received three before the deliberate assessment cap. The new mechanism correctly left the stock decision in charge. Even the broader weak-incumbent rule examined one real marginal window without selecting a replacement, and reproduced the prepared stock controls exactly through the end of the test. [Live actor controls and scoring](exact.json.gz) · [Weak-incumbent control](weak-exact.json.gz)

## Actual low returns after the body is ready

Using two paid preparation prefixes, the stock-window options returned **3/3** on identical physically received low balls and the unchanged opponent answered all three. Stock without the options scored **0/3** from the same prepared bodies. These include a new-to-the-motion-field .964277 m projected low, a broad native TT37 left redraw received at the same actual far impact and near bounce in both arms. The options accepted the new branch at tick 4079 after the detected bounce, with 2/2 perturbation support. The first two are teaching-source diagnostics. All comparisons in this section begin after a disclosed previously attained body posture, so they do not claim live preparation or a full game score. Zero clearance penetrations in 3,263 changed-tail CAD samples. [Exact native offers and resulting motor choices](low.json.gz)

For a separate prospective whole comparison, the optional `--weak` rule can consider a marginal stock incumbent. It waits deeper into a fresh post-bounce track, runs the rest of the current live references from the same body alongside learned alternatives, and keeps the incumbent if its recheck supports both perturbations. An alternative must improve the perturbed score and clear a larger modeled net margin and forward velocity. The trigger, lease, and frequency of stock search do not change.

## Fresh complete points

The four preassigned releases used the same ordinary actuators and initial state, the same risk-guided partial preparation and 12 independently earned optical examples in addition to the two signed low seeds. Prepared stock and options each scored one return across all four points. Per-release scores were 0, 1, 0 and 0 for both, with bitwise identical controls. Four optional search windows reached the candidate screen. Two faced a marginal incumbent, and one alternative from a marginal window passed the nominal margin screen but did not win the perturbation check. None was selected. Natural low waveforms were evaluated on 7812, without a legal option from its normal body. This small unselected course exercised the integration and its fallback but provides no evidence of a full-game improvement. [Full native games](live.json.gz) · [Locked natural examples](../tt54_earned_low/first12.json)

The four unique complete tapes all reproduced exactly under the independent native and full CAD replay. There were zero penetrations in 2,973 samples. The measured minimum was 15.99 mm. [Strict replay and exact offer comparison](live-audit.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt52_stock_repertory plan
python -m scripts.experimental.tt52_stock_repertory exact
python -m scripts.experimental.tt52_stock_repertory_low
python -m scripts.experimental.tt52_stock_repertory exact --weak --out artifacts/tt52_stock_repertory/weak-exact.json.gz
python -m scripts.experimental.tt52_stock_repertory live --weak --natural artifacts/tt54_earned_low/first12.json
python -m scripts.experimental.tt52_stock_repertory_audit
python -m pytest -q tests/test_tt52_stock_repertory.py
```
