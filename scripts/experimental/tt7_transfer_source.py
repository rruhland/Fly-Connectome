"""Record camera timing evidence during the unchanged table-tennis practice course."""

from pathlib import Path
from tempfile import TemporaryDirectory

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_continuous import ContinualPlayer, PredictiveServo
from .tt6_continuous_trial import parser, run
from .tt7_transfer_timing import optical_likelihood


class RecordingServo(PredictiveServo):
    def __init__(self, fit, timing):
        super().__init__(fit, timing)
        self.likelihoods = []

    def observe(self, actor, pitch):
        observer = self.observer
        position = actor.position if actor is not None and actor.observed else None
        likelihood = optical_likelihood(position, observer.previous, observer.history, observer.offset, observer.noise)
        self.likelihoods.append(likelihood.tolist() if likelihood is not None else None)
        super().observe(actor, pitch)


class RecordingPlayer(ContinualPlayer):
    def _make_servo(self):
        return RecordingServo(self.adapter, self.timing_memory)

    def finish(self, reward):
        details = super().finish(reward)
        details['optical_likelihoods'] = self.servo.likelihoods
        return details


def compact(recorded, reference, *, points_per_visit=None):
    """Verify that adding observation logs preserved the original playing decisions."""
    old = _read(reference)
    visits = []
    for practice, expected in zip(recorded['visits'], old['visits'][:len(recorded['visits'])], strict=True):
        if (practice['install'], practice['probes']) != (expected['install'], expected['probes']):
            raise ValueError('source commissioning commands diverged')
        points = []
        for point, original in zip(practice['points'][:points_per_visit], expected['points'][:points_per_visit], strict=True):
            if (point['action_sha256'], point['timing'], point['public_reward']) != (
                    original['action_sha256'], original['timing'], original['public_reward']):
                raise ValueError(f'source action tape changed at practice seed {point["seed"]}')
            keys = ('seed', 'action_sha256', 'timing', 'optical_likelihoods')
            points.append({key: point[key] for key in keys})
        visits.append(dict(install=practice['install'], points=points))
    return dict(schema='tt7-optical-practice-v1', source_sha256=_hash(reference), visits=visits)


def main():
    cli = parser()
    cli.add_argument('--reference', type=Path, help='match the original practice tape and write only camera evidence')
    cli.add_argument('--evidence-per-visit', type=int, help='retain a matched prefix of the plays from each visit')
    config = cli.parse_args()
    if not config.predictive or config.mode != 'carry':
        raise ValueError('the source comparison uses the existing predictive carried player')

    def make_player(adapter, prior, factory, response, timing):
        return RecordingPlayer(adapter, prior, factory, response, timing, predictive=True)

    if config.reference is None:
        run(config, make_player=make_player)
    else:
        output = config.output
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory() as temporary:
            config.output = Path(temporary) / 'practice.json.gz'
            run(config, make_player=make_player)
            _write(output, compact(_read(config.output), config.reference, points_per_visit=config.evidence_per_visit))


if __name__ == '__main__':
    main()
