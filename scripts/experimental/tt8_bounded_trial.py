"""Compare receiving games with full-plan motor bounds and the original servo."""

from pathlib import Path
from time import perf_counter, process_time

from .tt4_transfer_trial import _hash, _metrics, _read, _write
from .tt6_continuous import ContinualPlayer
from .tt6_continuous_trial import parser, prepare_point, run
from .tt6_replay import replay
from .tt8_bounded import BoundedPlayer


def main():
    cli = parser()
    cli.description = __doc__
    cli.add_argument('--motor-plan', choices=('bounded', 'legacy'), required=True)
    args = cli.parse_args()
    if args.auto_servo or args.rally_response or args.shot_memory is not None:
        cli.error('use this comparison with the standard predictive receiving objective')
    args.predictive = True
    player = BoundedPlayer if args.motor_plan == 'bounded' else ContinualPlayer
    source = {name: _hash(Path(__file__).with_name(name)) for name in (
        'tt6_bounded.py', 'tt6_continuous.py', 'tt6_continuous_trial.py', 'tt8_bounded.py', 'tt8_bounded_trial.py')}

    def make_player(*base):
        return player(*base, predictive=True, adjust=args.mode != 'frozen')

    began, cpu = perf_counter(), process_time()
    run(args, make_player=make_player)
    elapsed, consumed = perf_counter() - began, process_time() - cpu
    receipt = _read(args.output)
    receipt.update(motor_plan=args.motor_plan, wall_seconds=elapsed, cpu_seconds=consumed,
                   implementation_sha256=source)
    for visit in receipt['visits']:
        rows = visit['points']
        visit['summary'] = _metrics(rows)
        if args.motor_plan == 'bounded' and rows:
            visit['motor_plan'] = {key: sum(row['motor_plan'][key] for row in rows) for key in rows[0]['motor_plan']}
    replay(args.output, prepare=prepare_point)
    receipt['replayed_points'] = sum(len(visit['points']) for visit in receipt['visits'])
    _write(args.output, receipt)


if __name__ == '__main__':
    main()
