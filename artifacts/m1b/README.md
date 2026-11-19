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
