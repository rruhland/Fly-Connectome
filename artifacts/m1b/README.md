# Pong motor discovery checkpoints and receipts

`2026-11-19-motor-learning.zip` contains the experiment specifications, game measurements, numerical checkpoints, and a SHA-256 manifest. The readout and vision used in each experiment are identified in its specification. The branch includes the vision asset and all source needed for the visual button controller.

To resume the public button-effect learner from a saved state in the project's Python environment, run the following from the repository root. The acquired checkpoint follows the first 32 ordinary games of the independent qualification. A later checkpoint can be loaded from the `state` field of any saved game receipt.

```python
import json
import sys
from zipfile import ZipFile

sys.path[:0] = ["src", "scripts/experimental"]
from pong_visual_button_effects import VisualButtonEffects

with ZipFile("artifacts/m1b/2026-11-19-motor-learning.zip") as bundle:
    state = json.loads(bundle.read("button-qualification/acquired.json"))
learner = VisualButtonEffects.from_state(state)
```

The learner carries its measured effects, calibration progress, and random generator state between games. The experimental controller combines this state with the inherited visual interception policy. The [research report](../../docs/reconstruction/2026-11-19-motor-discovery-results.md) explains which components learned and which were already present.

## Carried Pong checkpoint

The transfer checkpoint includes motor calibration, the motor-order probabilities and delay estimate saved after restoration, and the score policy saved immediately after 96 chronological practice points. Its provenance includes controller hashes and the hash of the frozen visual model. To play one ordinary point using that checkpoint, run the following in Python from the repository root.

```python
import json
import sys
from pathlib import Path

sys.path[:0] = ["src", "scripts/experimental"]
from pong_score_placement import ScorePlacement
from pong_transfer_arena import play
from pong_risk_aware_placement import build_risk_aware_controller

state = json.loads(Path("artifacts/m1b/2026-11-19-pong-transfer-checkpoint.json").read_text())
scorer = ScorePlacement(**state["score"])
controller = build_risk_aware_controller(123, state["motor"], state["timing"], scorer, timing_updates=True)
point = play(controller, 1114851470, opponent_speed=.35, duration=12.)
print(point["hits"], point["score"])
```

For another point, construct a fresh controller with the same scorer object and `controller.timing.state()` in place of `state["timing"]`. That carries the learned order, lag, and score values while clearing the camera and physical queue for the new serve.
