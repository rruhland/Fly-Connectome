"""Regulation live-opponent protocols with one independent placement draw per reception."""

import numpy as np

from .tt29_partner_fast import NetRescuePracticePartner
from .tt31_curriculum_opponent import FIXED, CourseArena, placement


DIAGNOSTIC_CAP = 24
LONG_CAP = 200
MAX_CONTROL_STEPS = 24000
PROFILES = {
    'reach': ((-.28, .28), (-1.34, -1.22), (1.01, 1.11), (3.15, 3.98)),
    'lane': ((.17, .25), (-1.33, -1.25), (1.025, 1.10), (3.25, 3.90)),
}
PROTOCOLS = ('random', 'tt36', 'fixed')
PLACEMENT_FIELDS = ('x_m', 'near_y_m', 'z_m', 'toward_near_mps')


def draw(card, leg, protocol, legacy_rng):
    """Draw a goal, not a commanded ball state. Later draws use the reply seed."""
    index = card['index'] + leg
    mode = card.get('mode', 'full')
    opening_anchor = leg == 0 and card.get('opening') == 'anchor'
    if opening_anchor:
        return tuple(placement(mode, index, legacy_rng)), 'anchor'
    if protocol == 'fixed':
        return FIXED, 'fixed'
    reply_seed = int(card.get('reply_seed', card['seed']+710173))
    if protocol == 'tt36':
        rng = legacy_rng if leg==0 else np.random.default_rng([reply_seed,index])
        return tuple(placement(mode,index,rng)), 'tt36'
    bounds = np.asarray(PROFILES[card.get('profile','reach')])
    seed = [card['seed'],37117] if leg==0 else [reply_seed,leg,37131]
    return tuple(np.random.default_rng(seed).uniform(bounds[:,0],bounds[:,1])), 'random'


class LivePartner(NetRescuePracticePartner):
    """Finite-mass far paddle, driven using the current physical incoming flight."""

    def reset(self, **settings):
        super().reset(**settings)
        self.requests = []
        self.checked_bounce = None

    def _rescue(self):
        if self.arena.protocol != 'random':
            return super()._rescue()
        # An attempted rescue gets the same requested aim. No second draw or quiet switch to a fixed lane.
        a = self.arena
        now = float(a.data.time)
        q = a.data.qpos[self.qpos:self.qpos+3]
        v = a.data.qvel[self.qvel:self.qvel+3]
        stroke, note = self._plan(self.target, q, v, now, short_ball=True)
        self.reports.append(dict(time_s=now, stage='net_same_aim', **note))
        if stroke is not None:
            self.stroke = stroke

    def _refresh_bounce(self):
        a = self.arena
        if a.protocol!='random' or not a._received_bounce or self.stroke is None or not self.requests:
            return
        if self.requests[-1]['draw'] == 'anchor' or self.checked_bounce == self.reception or a._last_hitter != 1:
            return
        if self.requests[-1]['reception'] != self.reception:
            return
        the_table_touched = a._geom_ids['table'] in a._previous_touches
        if the_table_touched or a.velocity[1] <= .32:
            return
        self.checked_bounce = self.reception
        now = float(a.data.time)
        left = self.stroke.eta + self.stroke.start - now
        if left < .125:
            return
        previous = next((r for r in reversed(self.reports) if 'strike_xyz' in r), None)
        if previous is None:
            return
        trajectory = self.flight.path(a.ball,a.velocity,a.spin,-1,bounced=True,horizon=min(.95,left+.030))
        if not len(trajectory):
            return
        patch = trajectory[np.argmin(abs(trajectory[:,0]-left)),1:4]
        deviation = float(np.linalg.norm(patch-np.asarray(previous['strike_xyz'])))
        if deviation < .026:
            return
        q = a.data.qpos[self.qpos:self.qpos+3]
        v = a.data.qvel[self.qvel:self.qvel+3]
        stroke, note = self._plan(self.target,q,v,now,short_ball=a.ball[1]<.22 and a.velocity[1]<1.5)
        self.reports.append(dict(time_s=now,stage='bounce_refresh',deviation_m=deviation,replaced=stroke is not None,**note))
        if stroke is not None:
            self.stroke = stroke

    def advance(self):
        a = self.arena
        if a.card is not None and a._last_hitter==1 and a.velocity[1]>.32 and self.reception!=a.contacts:
            leg = len(self.requests)
            requested, origin = draw(a.card,leg,a.protocol,self.rng)
            self.targets = np.array([requested])
            self.requests.append(dict(leg=leg,target=list(requested),time_s=float(a.data.time),draw=origin,
                reception=int(a.contacts),observed_ball=a.ball.tolist(),observed_incoming=a.velocity.tolist(),
                observed_spin=a.spin.tolist(),carriage=a.data.qpos[self.qpos:self.qpos+7].tolist()))
        self._refresh_bounce()
        super().advance()


class LiveArena(CourseArena):
    """Only the far carriage sees placement goals or privileged ball state."""

    def __init__(self, *, protocol, cards, **settings):
        self.protocol = protocol
        self.launches = 0
        super().__init__(cards=cards, **settings)
        self.partner = LivePartner(self, targets=[FIXED], guard_table=True)

    def launch(self, shot, *, card=None):
        super().launch(shot,card=card)
        self.launches += 1

    def exchange(self, near, far):
        before = float(self.data.time)
        result = super().exchange(near,far)
        elapsed = float(self.data.time)-before
        if not (0 < elapsed <= self.control_dt+1e-8) or (
                not result[2] and not result[3] and abs(elapsed-self.control_dt)>1e-8):
            raise ValueError('Unexpected simulator time change during a point')
        return result


def arena(cards=(), *, protocol='random', **settings):
    """Drop-in course arena for direct RGB/encoder actors and legal referee callbacks."""
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown live-opponent protocol {protocol}')
    defaults = dict(strike_ready='lateral',width=384,height=288,clearance_guard=True,
                    fast_search=True,max_steps=MAX_CONTROL_STEPS,rate_scale=.8,velocity_guard=True,
                    motor_envelope=True,guard_table=True,net_rescue=True)
    return LiveArena(protocol=protocol,cards=cards,**(defaults|settings))
