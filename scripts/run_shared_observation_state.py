"""Run the unchanged camera audit with shared-observation association."""

from pathlib import Path

from camera_state_audit import main
from shared_observation_state import SharedObservationState


if __name__ == '__main__':
    main(state_type=SharedObservationState,
         out=Path('docs/experiments/2026-09-26-shared-observation-results.json'))
