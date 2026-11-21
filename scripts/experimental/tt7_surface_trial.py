"""Play through visually identical changes in physical table compliance."""
from .tt6_continuous_trial import parser, prepare_point, run
from .tt7_surface import SurfaceBelief, SurfacePlayer


SURFACE_PROTOCOL = 'changing-table-v1'
DAMPING = {'lively': 12., 'regular': 24., 'muted': 65., 'low': 90.}


def prepare_material(arena, phase):
    material, _, play = phase.partition('-')
    if material not in DAMPING:
        raise ValueError(f'unknown table material {material}')
    disturbance = prepare_point(arena, play or 'ordinary')
    arena.model.geom('table').solref[1] = -DAMPING[material]
    return disturbance


def main():
    p = parser()
    p.set_defaults(phases=['regular']*3 + ['muted']*4 + ['lively']*4 + ['low']*3 + ['regular']*2)
    p.add_argument('--read-only', action='store_true', help='measure bounces and retain legacy stroke planning')
    args = p.parse_args()
    if not args.predictive or args.mode != 'carry':
        p.error('this experiment uses the carried predictive receiver')
    belief = SurfaceBelief()

    def player(*inputs):
        return SurfacePlayer(*inputs, belief=belief, adaptive=not args.read_only, predictive=True)

    def finish(visit):
        visit['table_belief'] = dict(applied=not args.read_only, values=list(belief.values))

    run(args, make_player=player, after_visit=finish, prepare=prepare_material, protocol=SURFACE_PROTOCOL)


if __name__ == '__main__':
    main()
