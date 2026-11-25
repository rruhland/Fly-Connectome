"""Reproduce the exact frozen held protocol without changing the working tree."""

import argparse
import importlib.util
import sys
from pathlib import Path


def run(source, feeds, *, verify_only=False):
    if source is not None:
        # This module is compiled and imported under its ordinary package name, so
        # relative imports and the archived runner's own source receipt stay intact.
        source = source.resolve()
        if source.name != "tt42_learning_course.py":
            raise ValueError("Save the exact archived runner bytes as tt42_learning_course.py")
        name = __package__ + ".tt42_learning_course"
        spec = importlib.util.spec_from_file_location(name, source)
        course = importlib.util.module_from_spec(spec)
        sys.modules[name] = course
        spec.loader.exec_module(course)
        setattr(sys.modules[__package__], "tt42_learning_course", course)
    from . import tt42_direct_execution as e

    sources = (e.__file__, e.course.__file__, e.mpc.__file__, e.FIT, feeds,
               e.mpc.BANK, e.course.OPTIC, e.course.PRACTICE)
    protocol = e.HELD.with_suffix("").with_suffix(".protocol.json")
    expected = e._read(protocol)
    observed = {Path(p).name: e._hash(Path(p)) for p in sources}
    if expected != observed:
        different = sorted(k for k in expected.keys() | observed.keys() if expected.get(k) != observed.get(k))
        raise ValueError(f"Archived held protocol inputs changed: {', '.join(different)}")
    print("Verified exact held protocol", flush=True)
    if verify_only:
        return
    e.assess(e.HELD, deck="assessment", n=24, feeds=feeds, resume=e.HELD.exists())
    e.report()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--course-source", type=Path)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    run(a.course_source, a.feeds, verify_only=a.verify_only)
