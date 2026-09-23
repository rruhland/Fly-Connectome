"""Frozen downstream probe extended to measured LLPC2/3 and VS targets."""

from pathlib import Path

import motion_t5_downstream_probe as probe


if __name__ == '__main__':
    probe.TARGETS = probe.TARGETS + ('LLPC2', 'LLPC3', 'VS')
    probe.OUT = Path('docs/experiments/2026-09-23-t5-downstream-extension-results.json')
    probe.main()
