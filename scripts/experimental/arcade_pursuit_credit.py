"""Credit pursuit only when commanded paddle motion is supported by the camera."""

from dataclasses import asdict, dataclass

from pong_rapid_motor_recovery import COMMIT_THRESHOLD


PIXEL = 1 / 64
MIN_TARGET_GAP = .14
MIN_RESPONSE_SECONDS = .18
MAX_SAMPLE_GAP = .06


@dataclass(frozen=True)
class PursuitWitness:
    direction: str
    displacement: float
    elapsed: float
    arrival: float
    delay_commands: int
    order: tuple


class PursuitEvidence:
    """Associate a sustained own request with a visible near-term chase."""

    def __init__(self):
        self.run = None
        self.witness = None

    def record(self, seconds, detector, timing, request, intended):
        context = timing.arrival_context
        pair = detector.pair_ids
        belief = timing.belief
        if (pair is None or intended not in (0, 2) or context is None
                or belief.states is None or belief.probabilities[belief.confirmed] < COMMIT_THRESHOLD
                or timing.candidate_dwell or request != belief.order[intended]):
            self.run = None
            return
        observed = {row['id']: row['normalized_position'] for row in detector.canonical_entities
                    if row['observed']}
        if any(identity not in observed for identity in pair):
            self.run = None
            return
        paddle, ball = (observed[key][0] for key in pair)
        gap = context.intercept - paddle
        side = 1 if gap > 0 else -1
        # Aiming can legitimately leave the current ball. Require the forecast
        # and the visible ball to agree and leave room for a normal shot offset.
        if abs(gap) < MIN_TARGET_GAP or side * (ball - paddle) < 2 * PIXEL:
            self.run = None
            return

        key = (pair, belief.order, timing.delay_commands, request, intended, side)
        arrival = seconds + context.seconds
        if self.run is not None:
            old_key, started, previous, origin, target, deadline = self.run
        if (self.run is None or key != old_key or seconds - previous > MAX_SAMPLE_GAP
                or abs(context.intercept - target) > 3 * PIXEL or abs(arrival - deadline) > .18):
            self.run = (key, seconds, seconds, paddle, context.intercept, arrival)
            return
        self.run = (key, started, seconds, origin, target, deadline)
        elapsed = seconds - started
        lag = timing.delay_commands * timing.camera_seconds
        if elapsed < lag + MIN_RESPONSE_SECONDS:
            return
        displacement = paddle - origin
        command_direction = 1 if intended == 0 else -1
        if command_direction * displacement >= PIXEL:
            kind = 'toward' if side == command_direction else 'away'
            self.witness = PursuitWitness(kind, displacement, elapsed, arrival,
                                          timing.delay_commands, belief.order)

    def terminal_witness(self, seconds):
        """Select evidence for the incoming approach that ended this life."""
        event = self.witness
        if event is not None and -.10 <= seconds - event.arrival <= .30:
            receipt = asdict(event)
            receipt['order'] = list(receipt['order'])
            return receipt
        return None
