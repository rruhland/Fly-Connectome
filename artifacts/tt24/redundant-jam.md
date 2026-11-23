# Redundant wrist recovery after a mechanical jam

The original wrist-pitch hinge was locked at its own seeded reset angle with a MuJoCo joint equality constraint. The added oblique distal hinge stayed free in the recovery condition. An otherwise identical comparison locked both hinges. All six command inputs and encoder outputs stayed present in every condition, and the old motor could still exert torque against the constraint. [Physical fixture and fault](../../scripts/experimental/tt24_redundant.py)

The inherited five-joint wrist pupil retained its earlier teacher-trained perception and control. The previously measured stock action-effect map was frozen. Each normal or faulted receiving body got exactly **12 signed probe blocks in one reset**, with the same six channels, pulse order, and duration. This was 180 frames or 3.0 simulated seconds per body. The probes used own encoders and paddle landmarks from two fixed RGB cameras. The estimator used the five legacy encoder indices and reset angles as known correspondence. It fitted all six existing command channels without a fault label or a supplied route around the locked joint. Calibration was invoked after setting the fault configuration. Held control used the wrist camera and encoders. [Runner](../../scripts/experimental/tt24_redundant_jam.py)

The fitted original-pitch encoder response fell from 0.171 rad per normalized probe in the normal body to zero under the jam. The distal response remained 0.581 rad per normalized probe. The output coefficient from the inherited pitch command to the original actuator fell from 0.321 to zero, and the coefficient to the distal actuator rose from 0.929 to 1.039. Small cross-axis coefficients were also fitted. With both hinges locked, both direct encoder responses were zero at the reporting precision. [Measured action effects](redundant-jam.json.gz)

| Condition on 12 fresh paired drops | Total rebounds | Mean longest run | Best consecutive run |
| --- | ---: | ---: | ---: |
| Normal six-joint body, learned mapping | 107 | 7.67 | 13 |
| Original pitch jammed, distal hinge free, newly learned mapping | **89** | **7.08** | **14** |
| Original pitch jammed, frozen normal six-axis mapping | 78 | 6.08 | 14 |
| Original pitch jammed, frozen five commands padded with zero | 20 | 1.67 | 3 |
| Both pitch hinges jammed, newly learned mapping | 23 | 1.92 | 3 |
| Original pitch jammed, stationary | 31 | 2.58 | 4 |

Recovery with the distal hinge retained 83% of the normal total. Relative to both hinges locked, it gained 5.50 rebounds per drop and won eleven comparisons with one tie. The paired 95% drop-bootstrap interval was [3.17, 8.17]. Refitting after the jam added 0.92 rebounds per drop over the frozen normal six-axis map, with six gains, two ties, four losses, and a paired interval of [−2.25, 4.08]. The existing six-axis map already issued substantial distal commands before the fault. The recovered runs kept the original hinge within 0.000044 rad of reset and used a median distal peak excursion of 0.434 rad. The distal hinge was within 0.01 rad of its nominal stop or beyond it for a mean 2.6% of sampled frames. The original regularization was retained for this cohort. All 72 held command tapes reproduced the physical rebound events. [Scores, encoders and exact replay tapes](redundant-jam.json.gz)

A separate replay of fresh stock-derived optical endpoint goals used the same calibration data. Thirteen of sixteen goals had a complete marker record across treatments. The recovered map achieved 2.87 RMS pixel error versus 2.85 for normal and 6.85 for padded control. Both hinges jammed reached 2.09 and the frozen normal optical map applied under the jam reached 2.01. The powered distal hinge clearly improved the dynamic rebound task over the double jam, while the sampled endpoint goals favored the double-jammed map on mean pixel error. [Optical endpoint comparisons](redundant-jam.json.gz)
