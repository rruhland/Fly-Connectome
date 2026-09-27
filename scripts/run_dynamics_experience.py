"""Bounded training-diversity and causal online-learning comparison."""

from pathlib import Path

from frame_observation_state import FrameObservationState
from run_local_motion_dynamics import main


if __name__ == '__main__':
    for diverse, online, name in ((True, False, 'diverse-frozen'),
                                  (False, True, 'restricted-online'),
                                  (True, True, 'diverse-online')):
        main(state_type=FrameObservationState, diverse=diverse, online=online,
             out=Path(f'docs/experiments/2026-09-26-dynamics-{name}-results.json'))
