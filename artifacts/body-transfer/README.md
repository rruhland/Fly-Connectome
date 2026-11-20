# Cross-body reaching experiments

[Results and replays](../../docs/reconstruction/2026-11-20-cross-body-transfer.md) · [Compressed experiment records](cross-body-results.zip) · [SHA-256 manifest](receipt-sha256.json)

The archive contains the two-axis motor holdout, two channel-alignment runs, the source training and held-out evaluation of the exploration strategies, and the actual control sequences summarized by checksums and per-trial outcomes for the goal-transfer runs. The file with `held-event` in its name is an earlier diagnostic with an explicit physical contact event. The files containing `scalar` and `position-holdout` report ordinary per-tick score on two- and three-joint arms.

The `visual-utility/` files hold source practice, the frozen source weights and all four matched target controls. The `utility-surprise/` files contain the stable, inversion and independent-replacement courses. The `practice-curve` receipt records each independently evaluated practice dose from 0 to 24.

The separate JSON files below are acquired states from individual recorded source streams. The motor checkpoint carries an estimated mapping of commands and motion into visible displacement. The near/far checkpoint carries only the cue last associated with scalar reward. The visual utility checkpoint stores five fitted coefficients from source stream 17382002. Each loader starts a new instance.

```python
import json
from pathlib import Path
from body_effect_learner import BodyLearner
from body_strategy_transfer import RewardAssociation
from body_visual_utility import UtilityLearner

directory = Path("artifacts/body-transfer")
scorer = UtilityLearner(json.loads((directory / "carriage-visual-utility.json").read_text()))
arm = BodyLearner(prior=json.loads((directory / "nominal-arm-motor.json").read_text()))
preference = RewardAssociation.from_prior(
    json.loads((directory / "carriage-goal-association.json").read_text())
)
```

To reproduce the main experiments from the repository root with the project's Python environment and NumPy, use a new output directory. The following commands are for Bash.

```bash
export PYTHONPATH=src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p /tmp/body-transfer-results
python scripts/experimental/run_body_transfer.py /tmp/body-transfer-results/motor.json --seed 1758402197
python scripts/experimental/run_body_motor_alignment.py /tmp/body-transfer-results/alignment-a.json --seed 35422027
python scripts/experimental/run_body_motor_alignment.py /tmp/body-transfer-results/alignment-b.json --seed 87943019
python scripts/experimental/body_probe_strategy.py fit --directory /tmp/body-transfer-results/probe
python scripts/experimental/body_probe_strategy.py evaluate --directory /tmp/body-transfer-results/probe
python scripts/experimental/body_strategy_transfer.py --seeds {74100..74163} > /tmp/body-transfer-results/relations.json
python scripts/experimental/run_position_arm_transfer.py /tmp/body-transfer-results/position.json
python scripts/experimental/body_visual_utility.py fit --directory /tmp/body-transfer-results/utility
python scripts/experimental/body_visual_utility.py evaluate --directory /tmp/body-transfer-results/utility
python scripts/experimental/run_body_practice_curve.py /tmp/body-transfer-results/practice-curve.json
python scripts/experimental/body_utility_surprise.py fit --directory /tmp/body-transfer-results/surprise
python scripts/experimental/body_utility_surprise.py evaluate --directory /tmp/body-transfer-results/surprise
```
