# Earlier preparation when the stock stroke looks weak

The earlier impact review was very accurate on selected movements, but it mostly replaced already-playable shots. In physically matched starting states it did not add a legal contact, and the whole controller produced shorter rallies. This test asks whether the arm can prepare earlier from earned motor histories while leaving its existing stroke alone when it already looks viable. [Impact results and local rim conversion](../tt38/impact-margin.md)

The practice bank replays accepted independent impact-margin movements that actually landed and received a legal reply. The stored input is a stereo flight estimate and five joint encoder angles around 0.32 s before the predicted reception. The target is the arm's previously issued bounded joint reference sequence. The window and phase come from its camera ETA. Actual referee outcomes only label successful practice, and exact replay of the physical games checks the source tape. Release and outcome metadata are not inputs to movement selection. Six school assessment releases are disjoint from the independent practice releases.

At an early decision, the policy first tests the stock approach and a small stock correction deck in the same isolated optical model. It does not borrow a preparation if any stock option is nominally legal and passes at least three of five state settings. Otherwise it time-aligns up to four nearby earned motions, with a limited correction for the current arm geometry. It applies only the early segment and only if the private trajectory is nominally legal and passes at least four of five settings. At the ordinary late decision, it uses the updated RGB and joint history to test the borrowed suffix again. A viable stock action takes precedence. If the candidate failed the early screen, the point continues with the unmodified control. The physical table, broad independent opponent, and 200-return cap did not change.

The distinction between late action selection and physical preparation mattered on 3395. The early screen saw no nominal stock landing, although a stock approach survived three of five perturbed states. An earned preparation passed five of five and moved the arm at tick 417, after an exact matched command prefix. At tick 429 both the stock and borrowed final actions passed five of five. The arm used the stock action, but now from a different state. Its fourth stroke landed 13 cm shorter and slower, still legal and answered. The resulting point contained six arm returns rather than fourteen. A late decision to use stock does not move the arm back to where it would have been without the early switch.

## Result

The bank contained 16 real legal and answered motor histories from nine independent practice points. The six disjoint school points produced 51 legal arm returns with stock and 41 with protected preparation, with 108 and 88 total exchanges respectively. The longest points were 14 and 10 arm returns. Process CPU was 126.5 and 166.9 seconds. The per-point arm scores were 8→8, 14→6, 12→9, 13→10, 2→6 and 2→2. Later stock memory and replies could diverge after a different actual stroke.

Of 47 early reviews, 32 protected stock, 11 could not find a robust earned option, one did not have a nearby history, and three applied an early borrowed preparation. None of the three issued a borrowed strike at the late recheck. On 7228, even a preview with zero legal stock outcomes over all five state settings and four legal outcomes for the borrowed movement did not transfer the apparent advantage to the late optical check. We did not lower the gate or use these cases to choose a new threshold. The next useful test is to measure whether early optical previews predict *actual* late failure while leaving the arm's stock preparation unchanged. [Practice bank](margin-practice-bank.json.gz) · [Physical points](margin-practice-school.json.gz) · [Paired scores](margin-practice-school-summary.json)

## Reproduction

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt39_margin_practice collect artifacts/tt39/margin-practice-bank.json.gz artifacts/tt38/impact-margin-independent.json.gz
python -m scripts.experimental.tt39_margin_practice test artifacts/tt39/margin-practice-bank.json.gz artifacts/tt37/value-closed.json.gz --out artifacts/tt39/margin-practice-school.json.gz --n 6
python -m scripts.experimental.tt38_impact_margin artifacts/tt37/value-closed.json.gz artifacts/tt39/margin-practice-school-summary.json --compare artifacts/tt39/margin-practice-school.json.gz
```
