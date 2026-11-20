"""Use carried offensive placement only with a visible opposing paddle."""

from arcade_transfer_session import TransferSession


class VisualPlacementGate:
    def __init__(self, detector, placement_offset):
        self.detector = detector
        self.placement_offset = placement_offset
        self.opponent_frames = 0
        self.nonzero_offsets = 0

    def __call__(self, context):
        pair = self.detector.pair_ids
        if not self.detector.opponent_visible or pair is None:
            return 0.
        if not any(row['observed'] and row['id'] not in pair
                   for row in self.detector.canonical_entities):
            return 0.

        self.opponent_frames += 1
        offset = self.placement_offset(context)
        self.nonzero_offsets += int(bool(offset))
        return offset

    def state(self):
        return dict(opponent_frames=self.opponent_frames, nonzero_offsets=self.nonzero_offsets)


class GatedTransferSession(TransferSession):
    """Carry the same state and mask only the offensive offset callback."""

    def __init__(self, arm, policy_seed, checkpoint=None, *, initial_polarity=None):
        if arm not in ('full', 'fresh'):
            raise ValueError('the visual placement gate requires an active warm or reset learner')
        super().__init__(arm, policy_seed, checkpoint)
        if initial_polarity is not None:
            if arm != 'fresh' or initial_polarity not in (-1, 1):
                raise ValueError('only a reset arm can use a counterbalanced initial polarity')
            self.polarity = initial_polarity
            self.policy['choices'][4] = initial_polarity
        self.gate = None

    def begin_life(self):
        life = super().begin_life()
        timing = life.controller.timing
        self.gate = VisualPlacementGate(life.detector, timing.contact_offset)
        timing.contact_offset = self.gate
        return life

    def end_life(self):
        prior_shot_memory = self.shot_memory
        gate = self.gate
        super().end_life()
        if not gate.opponent_frames:
            # A life without a visible opponent supplies no pursuit evidence.
            self.shot_memory = prior_shot_memory
        self.gate = None
