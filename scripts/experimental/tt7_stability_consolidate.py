"""Extend a qualified receiver with subsequently identified motor maps."""

import argparse
from pathlib import Path

from .tt4_transfer_trial import _hash, _read, _write


def consolidate(qualified_path, experienced_path):
    qualified = _read(qualified_path)
    experienced = _read(experienced_path)
    schema = 'tt6-installation-memory-v1'
    if qualified['schema'] != schema or experienced['schema'] != schema:
        raise ValueError('expected portable installation memories')
    starting_models = qualified['models']
    new_models = experienced['models']
    if starting_models != new_models[:len(starting_models)]:
        raise ValueError('later practice must carry the earlier motor maps')
    return dict(qualified, models=new_models,
                consolidation=dict(qualified_sha256=_hash(qualified_path),
                                   experienced_sha256=_hash(experienced_path),
                                   retained_models=len(starting_models),
                                   added_models=len(new_models) - len(starting_models)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('qualified', type=Path)
    parser.add_argument('experienced', type=Path)
    args = parser.parse_args()
    _write(args.output, consolidate(args.qualified, args.experienced))
