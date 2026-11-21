"""Preserve the mini player's observed motor-response profiles through hall training."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write


def attach(checkpoint, source, output):
    """Restore source profiles and assign new, unsorted hall evidence to the shared prior."""
    state = _read(checkpoint)
    if 'motor_span' in state:
        raise ValueError('the checkpoint already carries response profiles')
    old = _read(source)['memory']
    if old['shared']['coefficients'] != state['numeric'][0]:
        raise ValueError('the source and receiving player use different spatial hypotheses')
    if not all((np.asarray(table['counts']) <= state['numeric'][2]).all() for table in old['profiles'].values()):
        raise ValueError('the supplied profiles contain evidence beyond the receiving checkpoint')
    # Preserve source-specific evidence wherever it differs from the shared
    # source. A bank that duplicated the shared source inherits the current
    # shared player when that response span is next observed.
    profiles = {k: v for k, v in old['profiles'].items() if v != old['shared']}
    shared = dict(zip(('coefficients', 'totals', 'counts'), state['numeric'], strict=True))
    state['motor_span'] = dict(shared=shared, profiles=profiles, active=None, exposures={})
    state['span_source_sha256'] = dict(player=_hash(checkpoint), earlier_profiles=_hash(source))
    _write(output, state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--profiles-from', type=Path, default=ROOT/'artifacts/tt4/mini-first-person-player.json.gz')
    a = p.parse_args()
    attach(a.checkpoint, a.profiles_from, a.output)
