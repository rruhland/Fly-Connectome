"""Execute a camera-selected, fixed workspace displacement in a scored two-arm point."""

import numpy as np

from .tt4_transfer_trial import _read
from .tt22_motor_transfer import MotorPrimitive
from .tt23_coach import PRACTICE
from .tt23_coach_trial import ReplyRanking
from .tt23_opening_motor import CHECKPOINT, described, select
from .tt23_opening_sequence import CorrectedOpeningStation
from .tt23_sequence import setup


class ExplicitJogStation(CorrectedOpeningStation):
    """A callback chooses a body-relative displacement before the common physical swing."""

    def __init__(self, choose_jog):
        common = setup()
        super().__init__(common['macro'], common['evidence'], common['far_profile'], _read(CHECKPOINT)['model'])
        self.choose_jog = choose_jog
        self.rival.confirm = ReplyRanking(self.rival, _read(PRACTICE)).confirm

    def plan(self, sight, mode, method):
        stock = MotorPrimitive(self.macro, None)
        goal, context = None, None
        if sight is not None and sight['mover'] is not None:
            jog = float(self.choose_jog(mode, sight, stock.ready_world.copy()))
            if not np.isfinite(jog) or abs(jog) > .271:
                raise ValueError('The selected jog is outside the trained motor range')
            goal = stock.ready_world[0] + .848 * jog
            goal, context = described(stock, sight, None, self.law, requested_goal=goal)
        option, parameters, error = select(stock, context, self.model, method)
        return goal, option, parameters, error
