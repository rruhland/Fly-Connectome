"""Run matched live receiving courses with a consolidated stroke memory."""

from pathlib import Path

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_continuous_trial import parser, run
from .tt7_stability import StableStrokePlayer, StrokeMemory
from .tt7_stability_residual import StrikeMemory, VisualStrikePlayer


def restore(path, practice, *, erase=False):
    state = _read(path)
    stored = state.get('stroke_memory')
    if stored is not None and practice:
        raise ValueError('the checkpoint already has stroke memory')
    memory = StrokeMemory.from_state(stored) if stored is not None else StrokeMemory(state['speed'])
    if erase:
        return StrokeMemory(memory.starting_speed)
    for source in practice:
        for visit in _read(source)['visits']:
            for point in visit['points']:
                if point['opportunity']:
                    memory.observe(point['impact']['used_speed'], point['public_reward'])
    return memory


def main():
    p = parser()
    p.description = __doc__
    p.set_defaults(resume=ROOT / 'artifacts/tt6/continual-player.json.gz',
                   predictive=True, cached_render=True, reflections=True)
    p.add_argument('--stroke-mode', choices=('stable', 'erased', 'legacy', 'residual', 'residual-erased'), default='stable')
    p.add_argument('--practice-receipts', type=Path, nargs='+', default=[])
    args = p.parse_args()
    if args.mode != 'carry' or args.auto_servo or args.rally_response or args.shot_memory or args.entry_memory:
        p.error('this trial uses carried predictive control')
    if args.stroke_mode == 'legacy':
        if args.practice_receipts:
            p.error('practice receipts require stroke memory')
        run(args)
        return
    residual = args.stroke_mode.startswith('residual')
    if (residual or args.stroke_mode == 'erased') and args.practice_receipts:
        p.error('practice receipts can initialize stable stroke memory only')
    memory_key = 'strike_memory' if residual else 'stroke_memory'
    if residual:
        saved = _read(args.resume).get(memory_key)
        memory = StrikeMemory.from_state(saved) if saved and args.stroke_mode == 'residual' else StrikeMemory()
    else:
        memory = restore(args.resume, args.practice_receipts, erase=args.stroke_mode == 'erased')
    source_hashes = {path.name: _hash(path) for path in args.practice_receipts}
    index = args.start_visit

    def make(*inputs):
        if residual:
            return VisualStrikePlayer(*inputs, strike_memory=memory)
        return StableStrokePlayer(*inputs, stroke_memory=memory)

    def record(visit):
        nonlocal index
        index += 1
        visit['stability'] = dict(mode=args.stroke_mode, practice_sha256=source_hashes, **{memory_key: memory.state()})
        if args.checkpoints:
            path = args.checkpoints / f'visit-{index}.json.gz'
            _write(path, dict(**_read(path), **{memory_key: memory.state()}))

    run(args, make_player=make, after_visit=record)


if __name__ == '__main__':
    main()
