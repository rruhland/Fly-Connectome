"""Compare a fixed and an online-selected rotary stroke under recurring conditions."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _hash, _metrics, _point, _write, load_sources
from .tt6_contact import AdaptiveContactPlayer, BASELINE_CONTACT, ContactPlayer, HIGH_LOFT
from .tt6_continuous_trial import point_condition
from .tt6_racket import set_racket_response
from .tt6_rotary_adapt_trial import commission
from .tt6_timing import DelayedWiring


PHASES = (('ordinary',) * 2 + ('soft',) * 4 + ('mixed',) * 5) * 2


def run(output, seed):
    geometry = dict(venue='hall', width=192, height=144, baseline=.28, max_steps=220)
    native, states, hashes = load_sources(ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz')
    arena, wiring, fit, factory, commissioning = commission(seed, len(PHASES), geometry)
    report = dict(schema='tt6-blocked-rotary-stroke-v1', seed=seed, geometry=geometry,
                  phases=PHASES, renderer='cached native with reflections', commissioning=commissioning,
                  source_sha256=hashes, runs={}, profiles=[dict(stroke=asdict(stroke), follow=asdict(follow))
                                                         for stroke, follow in (BASELINE_CONTACT, HIGH_LOFT)],
                  implementation_sha256={path.name: _hash(path) for path in
                                         (Path(__file__), Path(__file__).with_name('tt6_contact.py'))})
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for condition in ('fixed', 'adaptive'):
            prior = TransferPrior('first_person_mini', native, seed, mini=states['first_person_mini'])
            if condition == 'fixed':
                stroke, follow = BASELINE_CONTACT
                player = ContactPlayer(fit, prior, factory, stroke=stroke, follow=follow)
            else:
                player = AdaptiveContactPlayer(fit, prior, factory)
            rows = []
            section = dict(points=rows, summary={})
            report['runs'][condition] = section
            for index, (phase, start) in enumerate(zip(PHASES, commissioning['schedule']['tests'], strict=True)):
                response, disturbance = point_condition(phase)
                set_racket_response(arena, response)
                disturbed = DelayedWiring(wiring, disturbance, seed=seed)
                row = _point(disturbed, player, start)
                row.update(_counts(row), phase=phase, cycle=index // (len(PHASES) // 2), applied=disturbed.applied)
                rows.append(row)
                section['summary'] = {name: _metrics([r for r in rows if r['phase'] == name])
                                      for name in ('ordinary', 'soft', 'mixed')}
                _write(output, report)
                print(condition, index, phase, start, row['contacts'], row['player_returns'], row['rival_returns'],
                      row.get('profile_index'), row['evaluator']['reason'], flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=158747803)
    args = parser.parse_args()
    run(args.output, args.seed)
