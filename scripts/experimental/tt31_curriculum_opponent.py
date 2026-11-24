"""Live partner placements for courses on the regulation SO-101 court."""

import numpy as np

from .tt29_partner_fast import FastPartnerArena, NetRescuePracticePartner


FIXED = (0.235, -1.296, 1.055, 3.55)
LEFT = (-0.205, -1.296, 1.055, 3.55)
CENTER_DEPTH = -1.10
SIDE_THRESHOLD = 0.075


def placement(mode, index, rng):
    if mode == "fixed":
        return FIXED
    if mode == "periodic":
        return (FIXED, LEFT)[index % 2]
    if mode == "full":
        return rng.uniform(-.28, .28), rng.uniform(-1.34, -1.22), rng.uniform(1.01, 1.11), rng.uniform(3.15, 3.98)
    if mode != "continuous":
        raise ValueError(f"Unknown placement mode {mode}")
    # Visit both sides in a random order. Every coordinate is independently
    # drawn from a continuous interval. The point seed defines the choices.
    side = rng.choice((-1, 1))
    x = side * rng.uniform(0.13, 0.265)
    return x, rng.uniform(-1.34, -1.22), rng.uniform(1.01, 1.11), rng.uniform(3.15, 3.98)


def lane(x):
    if x <= -SIDE_THRESHOLD:
        return "left"
    if x >= SIDE_THRESHOLD:
        return "right"
    return "middle"


class CoursePartner(NetRescuePracticePartner):
    """Only the far paddle has access to placement goals and exact ball state."""

    def reset(self, **settings):
        super().reset(**settings)
        self.course_index = 0
        self.requests = []

    def advance(self):
        a = self.arena
        if a._last_hitter == 1 and a.velocity[1] > .32 and self.reception != a.contacts:
            card = a.card
            index = card["index"] + self.course_index
            requested = placement(card["mode"], index, self.rng)
            self.targets = np.array([requested])
            self.requests.append(dict(leg=self.course_index, target=list(requested), time_s=float(a.data.time)))
            self.course_index += 1
        super().advance()


class CourseArena(FastPartnerArena):
    """Start with one measured flight toward the far paddle and log actual replies."""

    def __init__(self, *, cards, **settings):
        self.cards = iter(cards)
        self.card = None
        super().__init__(**settings)
        self.partner = CoursePartner(self, targets=[FIXED], guard_table=True)

    def launch(self, shot, *, card=None):
        self.card = next(self.cards) if card is None else card
        if self.card["id"] != shot["id"]:
            raise ValueError("Course card and measured launch differ")
        if shot["pos_y"] >= 0 or shot["vel_y"] <= 0:
            raise ValueError("The measured opening must travel to the far paddle")
        self.crossings = {}
        self.before_step = None
        super().launch(shot)
        self._last_hitter = 1

    def _aerodynamics(self):
        if self._last_hitter == -1 and self._received_bounce:
            self.before_step = self.ball.copy()
        else:
            self.before_step = None
        super()._aerodynamics()

    def _contacts(self, incoming):
        b = self.before_step
        now = self.ball
        if b is not None and b[1] > CENTER_DEPTH >= now[1] and self.rival_returns not in self.crossings:
            fraction = (CENTER_DEPTH - b[1]) / (now[1] - b[1])
            crossing = b + (now - b) * fraction
            self.crossings[self.rival_returns] = dict(xyz=crossing.tolist(), velocity=list(incoming))
        super()._contacts(incoming)

    def point_placements(self):
        replies = []
        far_contact = None
        near = None
        for event in self.events:
            if event["kind"] == "contact" and event["side"] == -1:
                far_contact = event
                near = None
            elif event["kind"] == "legal_return" and event["side"] == -1:
                leg = len(replies)
                near = dict(leg=leg, target=self.partner.requests[leg]["target"],
                            far_contact=far_contact, near_bounce=event,
                            receive_line=self.crossings.get(leg + 1), near_contact=None,
                            near_landed=False)
                replies.append(near)
            elif near is not None and event["kind"] == "contact" and event["side"] == 1:
                if near["near_contact"] is None:
                    near["near_contact"] = event
            elif near is not None and event["kind"] == "legal_return" and event["side"] == 1:
                near["near_landed"] = True
        for reply in replies:
            if reply["receive_line"] is not None:
                reply["observed_lane"] = lane(reply["receive_line"]["xyz"][0])
            elif reply["near_contact"] is not None:
                reply["observed_lane"] = lane(reply["near_contact"]["ball"][0])
            else:
                reply["observed_lane"] = None
        return dict(card=self.card, requested=self.partner.requests, delivered=replies)


def replay_arena(cards, **settings):
    """Use the same flags as learning, with no camera or policy needed for tape replay."""
    defaults = dict(strike_ready="lateral", width=384, height=288, clearance_guard=True, fast_search=True,
                    max_steps=9000, rate_scale=.8, velocity_guard=True,
                    motor_envelope=True, guard_table=True, net_rescue=True)
    return CourseArena(cards=cards, **(defaults | settings))
