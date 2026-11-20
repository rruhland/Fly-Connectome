"""Learn when to probe on carriages, then freeze the choice on unfamiliar arms."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

import body_reaching_arena as reaching


SOURCE_TRAIN_SEED = 42159011
SOURCE_HOLDOUT_SEED = 62461909
SHUFFLE_SEED = 74599179
TARGET_SEED = 91346683
BOOTSTRAP_SEED = 62831853
SOURCE_TRAIN_CASES = 160
SOURCE_HOLDOUT_CASES = 96
TARGET_CASES = 64
TARGET_BODIES = ('arm', 'arm_variant', 'arm_remapped')
HORIZON = reaching.DEFAULT_MAX_STEPS
BOOTSTRAP_REPLICATES = 4000

FIRST_AXIS = (1., 0.)
SECOND_AXIS = (0., 1.)
STARTUP_PULSES = {
    'none': (),
    'one_axis': (FIRST_AXIS,),
    'two_axes': (FIRST_AXIS, SECOND_AXIS),
    'antithetic': (FIRST_AXIS, (-1., 0.), SECOND_AXIS, (0., -1.)),
}
BONUSES = (.55, .20, .95, 1.50)
BASELINE_BONUS = BONUSES[0]


@dataclass(frozen=True)
class Strategy:
    startup: str
    bonus: float


STRATEGIES = tuple(Strategy(startup, bonus) for startup in STARTUP_PULSES for bonus in BONUSES)

angles = np.arange(8) * (np.pi / 4)
unit_directions = np.round(np.column_stack((np.cos(angles), np.sin(angles))), 12)
ACTIONS = np.vstack((unit_directions, .5 * unit_directions, .25 * unit_directions, (0., 0.)))
ACTION_COSTS = .001 * np.sum(ACTIONS * ACTIONS, axis=1)


class ResponseEstimate:
    """Discounted local motor-to-pixel-displacement fit reset on every episode."""

    def __init__(self):
        self.information = np.zeros((2, 2))
        self.cross = np.zeros((2, 2))

    def observe(self, command, displacement):
        self.information *= .86
        self.information += np.outer(command, command)
        self.cross *= .86
        self.cross += np.outer(command, displacement)

    def choose(self, error, bonus):
        distance = float(np.linalg.norm(error))
        if distance <= .037:
            return ACTIONS[-1], False

        # Use the explicit inverse for a regularized symmetric two-channel fit.
        diagonal_a = self.information[0, 0] + .10
        diagonal_b = self.information[1, 1] + .10
        off_diagonal = self.information[0, 1]
        determinant = diagonal_a * diagonal_b - off_diagonal ** 2
        inverse = np.array(((diagonal_b, -off_diagonal), (-off_diagonal, diagonal_a))) / determinant
        weighted_actions = ACTIONS @ inverse
        predictions = weighted_actions @ self.cross
        remaining = error - predictions
        costs = np.sqrt(np.sum(remaining ** 2, axis=1)) + ACTION_COSTS
        exploit_index = int(np.argmin(costs))
        leverage = np.sum(weighted_actions * ACTIONS, axis=1)
        speed = max(.020, float(np.sqrt(np.max(np.sum(predictions ** 2, axis=1)))))
        scale = min(1., distance / .15) * speed
        uncertainty = scale * np.sqrt(leverage / (1. + leverage))
        index = int(np.argmin(costs - bonus * uncertainty))
        return ACTIONS[index], index != exploit_index


@dataclass(frozen=True)
class Outcome:
    startup_commands: int
    task_commands: int
    task_commands_changed_by_bonus: int
    total_commands: int
    held_goal: bool
    commands_to_held_goal_or_miss: int
    final_normalized_error: float
    weighted_normalized_error: float


def run_episode(arena, strategy, source_mix=None):
    """Broker camera markers and count every command, including hidden-goal pulses."""
    response = ResponseEstimate()
    startup_commands = 0
    task_bonus_commands = 0
    tip, goal = arena.observation()

    def step(command, previous_tip):
        physical_command = command if source_mix is None else source_mix @ command
        next_tip, next_goal = arena.step(physical_command)
        response.observe(command, next_tip - previous_tip)
        return next_tip, next_goal

    # The calibration schedule has no access to any goal reading.
    for pulse in STARTUP_PULSES[strategy.startup]:
        if arena.done:
            break
        tip, goal = step(np.asarray(pulse), tip)
        startup_commands += 1

    while not arena.done:
        command, changed_by_bonus = response.choose(goal - tip, strategy.bonus)
        tip, goal = step(command, tip)
        task_bonus_commands += changed_by_bonus

    return Outcome(
        startup_commands=startup_commands,
        task_commands=arena.steps - startup_commands,
        task_commands_changed_by_bonus=task_bonus_commands,
        total_commands=arena.steps,
        held_goal=arena.success,
        commands_to_held_goal_or_miss=arena.steps if arena.success else arena.max_steps + 1,
        final_normalized_error=arena.normalized_error,
        weighted_normalized_error=arena.weighted_normalized_error,
    )


@dataclass(frozen=True)
class SourceScene:
    body: str
    case: reaching.ReachCase
    action_mix: np.ndarray


def source_scenes(seed, count):
    rng = np.random.default_rng(seed)
    for index in range(count):
        case = reaching.sample_case(rng)
        angle = rng.uniform(0., 2 * np.pi)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        gains = rng.uniform(.60, 1., size=2)
        signs = rng.choice((-1., 1.), size=2)
        permutation = np.eye(2)[:, rng.permutation(2)]
        mixing = rotation @ permutation @ np.diag(gains * signs)
        body = ('carriage', 'carriage_variant')[index % 2]
        yield SourceScene(body, case, mixing)


def fit_strategy(costs):
    """Select the lowest factual mean elapsed-time cost with a fixed tie break."""
    return STRATEGIES[int(np.argmin(np.mean(costs, axis=0)))]


def shuffled_costs(costs, seed):
    """Destroy the within-scene correspondence between procedure and payoff."""
    rng = np.random.default_rng(seed)
    return np.array([rng.permutation(row) for row in costs])


def strategies_to_compare(learned, shuffled):
    return {
        'learned': learned,
        'zero_strategy_practice': Strategy('none', BASELINE_BONUS),
        'fixed_two_axes': Strategy('two_axes', BASELINE_BONUS),
        'fixed_antithetic_axes': Strategy('antithetic', BASELINE_BONUS),
        'source_payoff_shuffled': shuffled,
    }


def evaluate_source(seed, count, strategies):
    results = {label: [] for label in strategies}
    for scene in source_scenes(seed, count):
        for label, strategy in strategies.items():
            arena = reaching.ReachingArena(scene.body, scene.case, max_steps=HORIZON)
            results[label].append(run_episode(arena, strategy, scene.action_mix))
    return results


def confidence_interval(values, indices):
    return np.quantile(values[indices].mean(axis=1), (.025, .975)).tolist()


def summarize(results, cluster_size=1):
    """Bootstrap paired case indices, averaging each case's matched body variants."""
    command_costs = {
        label: np.array([result.commands_to_held_goal_or_miss for result in rows])
        .reshape(-1, cluster_size).mean(axis=1)
        for label, rows in results.items()
    }
    successes = {
        label: np.array([result.held_goal for result in rows], dtype=float)
        .reshape(-1, cluster_size).mean(axis=1)
        for label, rows in results.items()
    }
    count = len(command_costs['learned'])
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, count, size=(BOOTSTRAP_REPLICATES, count)
    )
    episode_metrics = {}
    contrasts = {}
    for label, rows in results.items():
        metrics = {}
        for field, _ in vars(rows[0]).items():
            metrics[f'mean_{field}'] = float(np.mean([getattr(row, field) for row in rows]))
        metrics['episode_count'] = len(rows)
        metrics['mean_commands_to_held_goal_or_miss_ci95'] = confidence_interval(
            command_costs[label], indices
        )
        metrics['held_goal_rate_ci95'] = confidence_interval(successes[label], indices)
        episode_metrics[label] = metrics
        if label != 'learned':
            savings = command_costs[label] - command_costs['learned']
            success_delta = successes['learned'] - successes[label]
            contrasts[label] = {
                'commands_saved_by_learned': float(np.mean(savings)),
                'commands_saved_by_learned_ci95': confidence_interval(savings, indices),
                'held_goal_rate_difference': float(np.mean(success_delta)),
                'held_goal_rate_difference_ci95': confidence_interval(success_delta, indices),
            }
    return {'conditions': episode_metrics, 'paired_contrasts': contrasts}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_signature():
    return {
        'runner_sha256': fingerprint(Path(__file__)),
        'arena_sha256': fingerprint(Path(reaching.__file__)),
    }


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fit(output):
    output.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    labels = {str(index): strategy for index, strategy in enumerate(STRATEGIES)}
    train = evaluate_source(SOURCE_TRAIN_SEED, SOURCE_TRAIN_CASES, labels)
    costs = np.array([[row.commands_to_held_goal_or_miss for row in train[label]]
                      for label in labels]).T
    shuffled = shuffled_costs(costs, SHUFFLE_SEED)
    chosen = strategies_to_compare(fit_strategy(costs), fit_strategy(shuffled))
    holdout = evaluate_source(SOURCE_HOLDOUT_SEED, SOURCE_HOLDOUT_CASES, chosen)
    report = {
        'signature': code_signature(),
        'numpy': np.__version__,
        'objective': 'Mean commands to four consecutive on-goal ticks, including all startup commands. A miss costs horizon plus one.',
        'horizon': HORIZON,
        'tick_seconds': reaching.TICK_SECONDS,
        'source_train_seed': SOURCE_TRAIN_SEED,
        'source_holdout_seed': SOURCE_HOLDOUT_SEED,
        'source_shuffle_seed': SHUFFLE_SEED,
        'source_train_cases': SOURCE_TRAIN_CASES,
        'source_holdout_cases': SOURCE_HOLDOUT_CASES,
        'source_train_policy_trials': SOURCE_TRAIN_CASES * len(STRATEGIES),
        'source_train_total_commands': sum(row.total_commands for rows in train.values() for row in rows),
        'source_holdout_total_commands': sum(row.total_commands for rows in holdout.values() for row in rows),
        'candidate_training_returns': [
            {'strategy': asdict(strategy), 'mean_cost': float(costs[:, index].mean()),
             'shuffled_mean_cost': float(shuffled[:, index].mean())}
            for index, strategy in enumerate(STRATEGIES)
        ],
        'source_training_costs_by_case_and_candidate': costs.tolist(),
        'source_heldout': summarize(holdout),
        'source_heldout_episodes': {label: [asdict(row) for row in rows] for label, rows in holdout.items()},
        'elapsed_seconds': monotonic() - started,
    }
    write_json(output / 'source.json', report)
    write_json(output / 'frozen.json', {
        'signature': code_signature(),
        'source_report_sha256': fingerprint(output / 'source.json'),
        'strategies': {label: asdict(strategy) for label, strategy in chosen.items()},
    })
    return chosen


def evaluate_target(output):
    freeze_path = output / 'frozen.json'
    lock = json.loads(freeze_path.read_text())
    if lock['signature'] != code_signature():
        raise ValueError('source and target evaluators have different code')
    if lock['source_report_sha256'] != fingerprint(output / 'source.json'):
        raise ValueError('source report differs from the frozen source report')
    started = monotonic()
    strategies = {label: Strategy(**fields) for label, fields in lock['strategies'].items()}
    seed = np.random.default_rng(TARGET_SEED)
    cases = [reaching.sample_case(seed) for _ in range(TARGET_CASES)]
    results = {}
    for body in TARGET_BODIES:
        results[body] = {
            label: [run_episode(reaching.ReachingArena(body, case, max_steps=HORIZON), strategy)
                    for case in cases]
            for label, strategy in strategies.items()
        }
    pooled = {
        label: [row for case_rows in zip(*(results[body][label] for body in TARGET_BODIES))
                for row in case_rows]
        for label in strategies
    }
    report = {
        'frozen_sha256': fingerprint(freeze_path),
        'target_seed': TARGET_SEED,
        'independent_cases_per_body': TARGET_CASES,
        'matched_case_design': 'Each arm and strategy uses the same independent held-out start/goal pairs.',
        'total_target_commands': sum(row.total_commands for arms in results.values()
                                     for rows in arms.values() for row in rows),
        'all_arms_case_clustered': summarize(pooled, cluster_size=len(TARGET_BODIES)),
        'per_body': {body: summarize(results[body]) for body in TARGET_BODIES},
        'cases': [asdict(case) for case in cases],
        'episodes': {
            body: {label: [asdict(row) for row in rows] for label, rows in arms.items()}
            for body, arms in results.items()
        },
        'elapsed_seconds': monotonic() - started,
    }
    write_json(output / 'target.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'evaluate'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'fit':
        print(json.dumps({label: asdict(strategy) for label, strategy in fit(args.directory).items()}))
    else:
        report = evaluate_target(args.directory)
        print(json.dumps({body: rows['paired_contrasts'] for body, rows in report['per_body'].items()}))


if __name__ == '__main__':
    main()
