# Separate local dynamics capacity test

Generic patch associations now pass both positional and appearance-coded
context relations under the declared current-image sensor. Run a separate
forecast-capacity experiment while the stronger camera-state audit completes.
This does not waive the still-failing event-only gate or establish production
readiness by itself.

Freeze the observed-region tracking front end. Learn next observed displacement
from four recent observed displacements with a bounded competitive associative
memory (128 prototypes). All training targets come from subsequently observed
region positions, not simulator trajectories. A generic local reference frame
derived from recent displacement makes the code rotation/scale comparable; no
direction classes, object names, boundary or collision rules are supplied.
Prototype and output updates use local running averages; no backpropagation.

Train on straight periodic movement and curved movement, then test new start
positions, shapes, speed scales and phases. Forecast autonomously at 1,4,8 camera
samples with no intervening image access. Include persistence, constant velocity,
constant acceleration, lag-two motion replay, and shuffled local associations.
Require at least a 20% mean endpoint-error reduction over constant velocity and
shuffled associations at horizons 4 and 8, without a material regression against
the best fixed temporal control. Report each scene family and tracking coverage;
do not remove untracked scenes or unfavorable horizons. This is a bounded
learned-dynamics test, not proof of general physics or calibrated uncertainty.
