"""Unchanged camera and dynamics audits with full-frame observation inference."""

from pathlib import Path

from camera_state_audit import main as camera_audit
from frame_observation_state import FrameObservationState
from run_local_motion_dynamics import main as dynamics_audit


if __name__ == '__main__':
    camera_audit(state_type=FrameObservationState,
                 out=Path('docs/experiments/2026-09-26-frame-observation-results.json'))
    dynamics_audit(state_type=FrameObservationState,
                   out=Path('docs/experiments/2026-09-26-frame-dynamics-results.json'))
