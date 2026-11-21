"""Package a game-learned SO-101 drive and replay the older addressed games."""

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .tt3_pong import state_sha256
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_retention import compare_gantry, compare_pong, gantry_spec, run_gantry, run_pong
from .tt20_curriculum import CHECKPOINT as FACTORY_SOURCE, OUTPUT as TT20, configuration, model_arena
from .tt20_fixture import TT20Arena
from .tt20_latency import CommandPipe
from .tt20_natural import NaturalViewTransport
from .tt21_game_delay import MAX_DELAY, GameDelayPlayer, NominalMotion, PassiveDelay
from .tt21_game_delay_trial import OUTPUT as SOURCE_GAME, play_point
from .tt21_game_eye_carry import OUTPUT as ARM_REFERENCE, source_checkpoint


TT19 = ROOT/'artifacts/tt19'
OUTPUT = ROOT/'artifacts/tt21'
LEGACY = TT19/'continual-so101-player.json.gz'
FACTORY = TT20/'curriculum-factory.json.gz'
ATLAS = TT20/'curriculum-dry-catalog.npz'
ADDRESS = 'tt21_so101_drive'
SCHEMA = 'tt21-addressed-drive-v1'
EYES = ('left', 'wrist')
TAPE_FIELDS = ('actions', 'applied', 'optical', 'rgb_sha256', 'outcome', 'events', 'reward', 'plans', 'lead')


def content(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=_hash(path))


def package(path):
    if path.resolve() in {asset.resolve() for asset in (LEGACY, SOURCE_GAME, FACTORY_SOURCE, FACTORY, ATLAS, ARM_REFERENCE)}:
        raise ValueError('the portable bundle needs its own path')
    state = _read(LEGACY)
    if ADDRESS in state:
        raise ValueError('the pre-TT21 checkpoint already has a drive address')
    source = _read(SOURCE_GAME)
    for key, asset in (('source_sha256', FACTORY_SOURCE), ('factory_sha256', FACTORY), ('atlas_sha256', ATLAS)):
        if source[key] != _hash(asset):
            raise ValueError(f'the SO-101 controller source changed at {asset.name}')
    programs, _, law, response, candidate = configuration(TT20)
    if int(candidate) != source['candidate']:
        raise ValueError('the saved drive trial used a different factory opening')
    drives = {}
    for eye in EYES:
        lineage = source_checkpoint(source, eye)
        game = next(pair['games']['adaptive'] for pair in source['results'][eye]
                    if pair['seed'] == lineage['game_seed'])
        learned = game['points'][lineage['first_changed_point']-1]['learning']
        drives[eye] = dict(delay=learned['delay'], loss=learned['carried_loss'], source=lineage)
    inherited = state_sha256(state)
    state[ADDRESS] = dict(schema=SCHEMA, inherited=content(LEGACY), inherited_state_sha256=inherited,
                          training=content(SOURCE_GAME), factory=content(FACTORY),
                          factory_source=content(FACTORY_SOURCE), atlas=content(ATLAS),
                          arena=source['arena'], candidate=int(candidate), programs=programs,
                          law=law.state(), response=response.tolist(), drives=drives)
    _write(path, state)
    return state[ADDRESS]


def address(path):
    state = _read(path)
    saved = state.pop(ADDRESS, None)
    if saved is None or saved['schema'] != SCHEMA or state_sha256(state) != saved['inherited_state_sha256']:
        raise ValueError('the older checkpoint fields changed or the drive schema is unsupported')
    return saved


class DriveSession:
    """Resume the selected eye's optic observer and persistent delay belief between points."""

    def __init__(self, saved, catalogue, eye, *, adaptive):
        self.eye = eye
        self.candidate = saved['candidate']
        drive = saved['drives'][eye]
        delay, loss = drive['delay'], np.asarray(drive['loss'], dtype=float)
        if type(delay) is not int or delay not in range(MAX_DELAY+1) or loss.shape != (MAX_DELAY+1,) or not np.isfinite(loss).all():
            raise ValueError('invalid saved drive state')
        self.arena = TT20Arena(**saved['arena'], view_mode=eye)
        law = FlightLaw.from_state(saved['law'])
        self.player = GameDelayPlayer(saved['programs'], catalogue, law, np.asarray(saved['response']), model_arena())
        self.player.lead = delay
        self.learner = PassiveDelay(NominalMotion(eye, saved['arena'])) if adaptive else None
        if self.learner is not None:
            self.learner.delay, self.learner.loss = delay, loss.copy()
        lenses = self.arena.lens_intrinsics()
        self.optics = NaturalViewTransport(eye, lens_fov={eye: lenses[eye]}, law=law)

    def point(self, seed, role, actuation):
        # The caller supplies the apparatus or evaluator transport. The drive
        # loader is independent of the true latency in that transport.
        return play_point(self.arena, self.player, self.optics, self.eye, seed,
                          role, self.candidate, actuation, self.learner)

    def close(self):
        if self.learner is not None:
            self.learner.nominal.close()
        self.player.player.close()
        self.arena.close()


@contextmanager
def resume_drive(path, eye, *, assets=ROOT, adaptive=True):
    """Load a fresh native point session without opening the donor or practice files."""
    saved = address(path)
    if eye not in EYES:
        raise ValueError('select left or wrist for the single-eye drive')
    atlas = assets/saved['atlas']['path']
    if _hash(atlas) != saved['atlas']['sha256']:
        raise ValueError('the public dry catalogue differs from the checkpoint')
    with np.load(atlas, allow_pickle=False) as archive:
        catalogue = dict(archive)
    session = DriveSession(saved, catalogue, eye, adaptive=adaptive)
    try:
        yield session
    finally:
        session.close()


def legacy(path, kind):
    saved = address(path)
    reference_path = TT19/f'retention-{kind}-after-factory.json.gz'
    reference = _read(reference_path)
    if reference['checkpoint_sha256'] != saved['inherited']['sha256']:
        raise ValueError('the pre-TT21 match used a different checkpoint')
    if kind == 'gantry':
        result = run_gantry(path, gantry_spec())
        comparison = compare_gantry(reference, result)
        count = comparison['matched_servers']
        if (not comparison['same_address'] or comparison['replayed_points'] != [count, count] or
                any(comparison[field] != count for field in ('exact_rgb_tapes', 'exact_action_tapes', 'exact_outcomes'))):
            raise ValueError('the old gantry game changed')
    else:
        import torch

        torch.set_num_threads(1)
        result = run_pong(path)
        comparison = compare_pong(reference, result)
        for name in ('routed', 'shared_only'):
            count = len(reference[name]['points'])
            if any(comparison[name][field] != count for field in ('exact_action_tapes', 'exact_outcomes')):
                raise ValueError(f'the old native Pong policy changed in {name}')
        if any(not comparison[key] for key in ('same_full_state', 'same_shared_state', 'same_rank_one_profile')):
            raise ValueError('the old Pong state changed')
    return dict(reference=content(reference_path), comparison=comparison, carried=result)


def replay_arm(path):
    saved = address(path)
    evidence = _read(ARM_REFERENCE)
    if evidence['arena'] != saved['arena']:
        raise ValueError('the reference arm starts came from a different source run')
    rows = {}
    for eye in EYES:
        destination = evidence['destinations'][eye]
        source = destination['source']
        if source != saved['drives'][source['eye']]['source']:
            raise ValueError('the exact source checkpoint for the paired eye differs')
        # Exercise a published legal receiving tape where this eye has one.
        # This is a serialization regression, with no new performance estimate.
        ref = next((point for point in destination['conditions']['carry'] if point['outcome']['legal_returns']),
                   destination['conditions']['carry'][0])
        with resume_drive(path, eye) as session:
            if session.learner.delay != ref['lead']:
                raise ValueError('the carried drive does not match the known scalar lead')
            start_loss = session.learner.loss.tolist()
            actual = session.point(ref['seed'], ref['role'], CommandPipe(destination['source']['evaluator_delay']))
            after = dict(delay=session.learner.delay, loss=session.learner.loss.tolist())
        matches = {key: state_sha256(actual[key]) == state_sha256(ref[key]) for key in TAPE_FIELDS}
        if not all(matches.values()):
            raise ValueError(f'the native restored point changed for {eye}: {matches}')
        rows[eye] = dict(reference_seed=ref['seed'], exact=matches, initial_loss=start_loss, next_drive=after,
                         point=actual)
    return dict(reference=content(ARM_REFERENCE), reference_training_sha256=evidence['source_sha256'],
                exact_source_exposure=True, carried_checkpoint_sha256=_hash(path), eyes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('build', 'gantry', 'pong', 'arm', 'point'))
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--assets', type=Path, default=ROOT, help='root containing the hash-pinned dry catalogue')
    parser.add_argument('--eye', choices=EYES, default='wrist')
    parser.add_argument('--seed', type=int, default=1428607538)
    parser.add_argument('--role', type=int, choices=(-1, 1), default=-1)
    parser.add_argument('--fixture-delay', type=int, default=0, help='simulated evaluator transport in control frames')
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    bundle = args.folder/'carry-checkpoint-player.json.gz'
    if args.stage == 'build':
        result = package(bundle)
        print('bundle', _hash(bundle), {eye: state['delay'] for eye, state in result['drives'].items()})
    elif args.stage == 'point':
        with resume_drive(bundle, args.eye, assets=args.assets) as session:
            result = session.point(args.seed, args.role, CommandPipe(args.fixture_delay))
        _write(args.folder/'carry-checkpoint-point.json.gz', result)
        print('point', result['seed'], result['lead'], result['reward'], result['outcome']['legal_returns'])
    else:
        result = replay_arm(bundle) if args.stage == 'arm' else legacy(bundle, args.stage)
        _write(args.folder/f'carry-checkpoint-{args.stage}.json.gz', result)
        if args.stage == 'arm':
            print('native SO-101', {key: (value['reference_seed'], value['point']['reward'],
                                          value['point']['outcome']['legal_returns']) for key, value in result['eyes'].items()})
        elif args.stage == 'pong':
            print('native Pong', {key: (result['comparison'][key]['exact_action_tapes'],
                                       result['comparison'][key]['exact_outcomes']) for key in ('routed', 'shared_only')})
        else:
            comparison = result['comparison']
            print('gantry', comparison['after']['score'], comparison['after']['total_legal_returns'],
                  comparison['exact_rgb_tapes'], comparison['exact_action_tapes'], comparison['exact_outcomes'])


if __name__ == '__main__':
    main()
