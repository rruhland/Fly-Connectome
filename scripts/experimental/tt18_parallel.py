"""Parallelize independent surface streams while keeping each learner sequential."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from .tt18_surface_report import compare
from .tt18_surface_trial import fit_rule, run


def benchmark(folder, workers):
    if workers < 1:
        raise ValueError('the number of workers must be positive')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        def batch(modes):
            futures = [pool.submit(run, folder, site, mode)
                       for mode in modes for site in ('gantry', 'five')]
            for future in futures:
                future.result()

        batch(('source',))
        fit_rule(folder)
        batch(('inherited', 'global', 'local', 'transfer'))
    return compare(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    benchmark(args.output, args.workers)
