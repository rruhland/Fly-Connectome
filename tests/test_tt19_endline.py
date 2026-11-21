import numpy as np
from scripts.experimental.tt4_arena import BALL_RADIUS
from scripts.experimental.tt19_so101 import SO101Arena


def test_long_ball_beyond_endline_and_legal_bounce():
    arena=SO101Arena(venue='hall',detail='preview',strict_endline=True)
    try:
        for hitter in (-1,1):
            arena.reset(seed=5,server=hitter,render=False)
            arena._waiting_for_strike=False
            arena._last_hitter=hitter
            direction=(0.,2.*hitter,0.)
            arena.ball[1]=hitter*(arena.table_half_length+.5*BALL_RADIUS)
            arena._contacts(direction)
            assert not arena.terminated
            arena.ball[1]=hitter*(arena.table_half_length+1.5*BALL_RADIUS)
            arena._received_bounce=True
            arena._contacts(direction)
            assert not arena.terminated
            arena._received_bounce=False
            arena._contacts(direction)
            assert arena.terminated and arena.reward==-hitter
            assert arena.reason=='long_past_endline'
    finally:
        arena.close()
