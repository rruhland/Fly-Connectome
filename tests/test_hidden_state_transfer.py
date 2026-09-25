"""Generic scene transfer scores causal latent forecasts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from correlation_latent_robustness import make_robust_cases
from learned_transition_units import TransitionPopulation
from local_hidden_transition import LocalHiddenTransition
from run_hidden_state_transfer import evaluate


def test_independent_scene_scores_next_state_against_persistence():
    models = {name: LocalHiddenTransition(
        TransitionPopulation(channels=16))
        for name in ('learned', 'shuffled_credit', 'frozen')}
    cases = [row for row in make_robust_cases() if row[0] == 'independent'][:1]
    result = evaluate(models, cases)
    assert result['visible']['independent']['learned']['frames'] == 10
    assert result['visible']['independent']['persistence']['frames'] == 10
