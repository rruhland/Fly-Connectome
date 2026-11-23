"""Render a verified learned table-tennis continuation from both issued motor tapes."""

import argparse

from .tt4_transfer_trial import _read
from . import tt23_sequence_coach_movie as movie
from .tt23_sequence import verify
from .tt24_rally_bank import RECEIPT as BANK
from .tt24_rally_from_release import FRESH as FROM_RELEASE
from .tt24_rally_reward import RECEIPT as PRACTICE
from .tt24_rally_trial import RECEIPT, returns


def render_held():
    data = _read(RECEIPT)
    if data['verified_tapes'] != sum(len(case['results']) for case in data['cases']):
        raise ValueError('the assessment has an unverified command tape')
    row = max((case['results']['demo_full'] for case in data['cases']), key=lambda item: item['longest'])
    movie.render_tape(row, RECEIPT, RECEIPT.with_name('rally-joint-feedback.mp4'), branch='demo_full',
                      subtitle='Near arm takes live RGB control after the shared opening',
                      terminal_caption='Complete point replayed from the issued joint rates')


def render_practice():
    data = _read(PRACTICE)
    row = next(entry['result'] for entry in data['practice'] if len(returns(entry['result'], 1)) >= 5)
    verify([dict(results={'reward': row})])
    width, height = data['metadata']['dimensions']
    movie.CONFIG = dict(movie.CONFIG, width=width, height=height)
    movie.render_tape(row, PRACTICE, PRACTICE.with_name('rally-reward-practice.mp4'), branch='practice',
                      subtitle='Reward-search practice after the shared teacher opening',
                      terminal_caption='Complete point replayed from the issued joint rates')


def render_from_release():
    data = _read(FROM_RELEASE)
    if data['verified_tapes'] != sum(len(case['results']) for case in data['cases']):
        raise ValueError('the from-release assessment has an unverified tape')
    row = max((case['results']['live'] for case in data['cases']), key=lambda item: item['longest'])
    width, height = data['metadata']['render_dims']
    movie.CONFIG = dict(movie.CONFIG, width=width, height=height)
    movie.render_tape(row, FROM_RELEASE, FROM_RELEASE.with_suffix('').with_suffix('.mp4'), branch='live',
                      subtitle='Near arm drives every frame from live RGB and joint readings',
                      terminal_caption='Complete point replayed from the issued joint rates')


def render_broad_bank():
    data = _read(BANK)
    if data['verified_tapes'] != sum(len(case['results']) for case in data['cases']):
        raise ValueError('the broader assessment has an unverified tape')
    row = max((case['results']['broad'] for case in data['cases']), key=lambda item: item['longest'])
    width, height = data['metadata']['dimensions']
    movie.CONFIG = dict(movie.CONFIG, width=width, height=height)
    movie.render_tape(row, BANK, BANK.with_suffix('').with_suffix('.mp4'), branch='broad',
                      subtitle='Broader-fit arm controls every frame from live RGB',
                      terminal_caption='Complete point replayed from the issued joint rates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--practice', action='store_true')
    mode.add_argument('--from-release', action='store_true')
    mode.add_argument('--bank', action='store_true')
    args = parser.parse_args()
    if args.practice:
        render_practice()
    elif args.from_release:
        render_from_release()
    elif args.bank:
        render_broad_bank()
    else:
        render_held()
