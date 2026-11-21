"""Measure hidden-source acquisition using motor experience from coupled table tennis."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read
from .tt14_gain_hardware import GainWiring
from .tt14_gain_trial import load_map
from .tt14_signal_search import trial


def run(folder, site, seed, count):
    priors, parents = {}, {}
    environment = None
    for name in ('fixed','refit','remap','hybrid'):
        prefix = folder / f'gain-{site}-coupling-{name}'
        game = _read(prefix.with_suffix('.json.gz'))
        checkpoint = prefix.with_name(prefix.name+'-changed-player.json.gz')
        digest = _hash(checkpoint)
        if digest != game['games'][0]['checkpoint']:
            raise ValueError('checkpoint is from a different game')
        setup = (game['arena'],game['matrix'])
        if environment is not None and setup != environment:
            raise ValueError('motor conditions differ between methods')
        environment = setup
        model, transform = load_map(checkpoint)
        priors[name] = model.jacobian(np.zeros(model.action_size))[:3] @ transform
        parents[name] = dict(checkpoint=digest,game_seed=game['games'][0]['seed'])
    settings, matrix = environment
    trial(priors,parents,settings,lambda arena: GainWiring(arena,settings['seed'],matrix),
          folder/f'signal-{site}-coupling.json.gz',seed,count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('site',choices=('gantry','five'))
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--count',default=12,type=int)
    a = parser.parse_args()
    run(a.folder,a.site,a.seed,a.count)
