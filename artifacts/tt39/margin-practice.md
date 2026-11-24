# Earlier preparation when the stock stroke looks weak

The earlier impact review was very accurate on selected movements, but it mostly replaced already-playable shots. In physically matched starting states it did not add a legal contact, and the whole controller produced shorter rallies. This test asks whether the arm can prepare earlier from earned motor histories while leaving its existing stroke alone when it already looks viable. [Impact results and local rim conversion](../tt38/impact-margin.md)

The practice bank replays accepted independent impact-margin movements that actually landed and received a legal reply. The stored input is a stereo flight estimate and five joint encoder angles around 0.32 s before the predicted reception. The target is the arm's previously issued bounded joint reference sequence. The window and phase come from its camera ETA. Actual referee outcomes only label successful practice, and exact replay of the physical games checks the source tape. Release and outcome metadata are not inputs to movement selection. Six school assessment releases are disjoint from the independent practice releases.

At an early decision, the policy first tests the stock approach and a small stock correction deck in the same isolated optical model. It does not borrow a preparation if any stock option is nominally legal and passes at least three of five state settings. Otherwise it time-aligns up to four nearby earned motions, with a limited correction for the current arm geometry. It applies only the early segment and only if the private trajectory is nominally legal and passes at least four of five settings. At the ordinary late decision, it uses the updated RGB and joint history to test the borrowed suffix again. A viable stock action takes precedence. If the candidate failed the early screen, the point continues with the unmodified control. The physical table, broad independent opponent, and 200-return cap did not change.

## Reproduction

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt39_margin_practice collect artifacts/tt39/margin-practice-bank.json.gz artifacts/tt38/impact-margin-independent.json.gz
python -m scripts.experimental.tt39_margin_practice test artifacts/tt39/margin-practice-bank.json.gz artifacts/tt37/value-closed.json.gz --out artifacts/tt39/margin-practice-school.json.gz --n 6
python -m scripts.experimental.tt38_impact_margin artifacts/tt37/value-closed.json.gz artifacts/tt39/margin-practice-school-summary.json --compare artifacts/tt39/margin-practice-school.json.gz
```
