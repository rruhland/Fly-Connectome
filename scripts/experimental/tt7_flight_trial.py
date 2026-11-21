"""Compare an optical outgoing-flight learner to its unchanged hall reference."""
from pathlib import Path

from .tt4_transfer_trial import _read, _write
from .tt6_continuous_trial import parser, run
from .tt7_flight import FlightMemory, FlightPlayer


def main():
    p = parser()
    p.add_argument('--feedback', choices=('adapt', 'fixed', 'observe', 'impulse', 'risk'), default='adapt')
    p.add_argument('--freeze-flight', action='store_true', help='hold the supplied optical calibration fixed')
    p.add_argument('--swing-velocity', action='store_true')
    p.add_argument('--flight-in', type=Path)
    p.add_argument('--flight-out', type=Path)
    args = p.parse_args()
    if not args.predictive or args.mode != 'carry':
        p.error('use predictive carried control for this comparison')
    if args.freeze_flight and not args.flight_in:
        p.error('freezing the optical model requires a saved flight state')
    saved = _read(args.flight_in) if args.flight_in else {}
    memory = FlightMemory(args.feedback, saved.get('errors', ()), saved.get('impulses', ()), frozen=args.freeze_flight)

    def make(*inputs):
        if args.swing_velocity:
            from .tt7_swing import SwingPlayer
            return SwingPlayer(*inputs, predictive=True, flight_memory=memory)
        return FlightPlayer(*inputs, predictive=True, flight_memory=memory)

    def record(visit):
        visit['flight_model'] = dict(mode=args.feedback, frozen=args.freeze_flight, swing_velocity=args.swing_velocity, errors=list(memory.history),
                                     impulses=list(memory.impulses))
        if args.flight_out:
            _write(args.flight_out, visit['flight_model'])

    run(args, make_player=make, after_visit=record)


if __name__ == '__main__':
    main()
