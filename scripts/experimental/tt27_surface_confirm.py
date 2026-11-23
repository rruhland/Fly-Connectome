"""Prospective test of the motor memory on four more unpracticed regulation-table surfaces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from .tt4_transfer_trial import _read, _write
from .tt27_context_memory import run as contextual
from .tt27_flight_feedback import OUT
from .tt27_surface_transfer import assess, block, HEIGHTS


def run(path):
    practice = [b for b in _read(OUT / "surface-transfer.json.gz")["blocks"] if b["label"] == "practice"]
    jobs = [(path, "held", i, s, 975014 + i, 48) for i, s in enumerate((2.32, 0.76, 1.48, 2.90))]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        blocks = practice + list(pool.map(block, jobs))
    receipt = OUT / "surface-confirmation.json.gz"
    _write(receipt, dict(heights=HEIGHTS.tolist(), blocks=blocks, assessment=assess(blocks)))
    contextual(receipt, OUT / "context-confirmation.json.gz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
