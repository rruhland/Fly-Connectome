"""Validate motor corrections on another scored point before using them."""

import numpy as np

from .tt14_gain_model import GainMemory, collect, fit_transform


def improvement(old, new):
    return min(a - b for a, b in zip(old, new, strict=True))


class GuardedMemory(GainMemory):
    """Use an entire later optical history to validate a map fitted on an earlier one."""

    def __init__(self, recalled):
        super().__init__(recalled.source, 'hybrid')
        self.maps = recalled.maps
        self.active = recalled.active

    def _changes(self, active, previous, new_errors, obs):
        for index, candidate in enumerate(self.maps):
            if index != self.active:
                yield index, candidate
        for older, errors in previous:
            if min(errors) < .050:
                continue
            for kind in ('gain', 'coupling'):
                candidate = fit_transform(self.source, [older], active.transform, kind)
                if candidate is not None and improvement(new_errors, candidate.errors(obs)) >= .022:
                    yield len(self.maps), candidate

    def observe(self, actions, trace):
        obs = collect(actions, trace, self.source.action_size)
        update = dict(previous=self.active, active=self.active, bank=len(self.maps))
        if min(obs.train.sum(), (~obs.train).sum()) < 5:
            return dict(update, reason='insufficient sightings')
        active = self.maps[self.active]
        errors = active.errors(obs)
        previous = [(older, active.errors(older)) for older in self.support]
        self.support = [*self.support[-1:], obs]
        update['errors'] = errors
        if not previous or min(errors) < .050:
            return dict(update, reason='waiting for corroborating motion')

        validated = []
        for index, candidate in self._changes(active, previous, errors, obs):
            candidate_errors = candidate.errors(obs)
            threshold = .015 if index == 0 else .022
            current_gain = improvement(errors, candidate_errors)
            if current_gain < threshold:
                continue
            prior_gains = [improvement(prior_errors, candidate.errors(past)) for past, prior_errors in previous]
            joint_gain = min(current_gain, max(prior_gains))
            if joint_gain < threshold:
                continue
            penalty = .004 if candidate.kind == 'coupling' and index >= len(self.maps) else 0.
            validated.append((joint_gain - penalty, index, candidate, candidate_errors, prior_gains))
        if not validated:
            return dict(update, reason='no independently supported map')

        _, index, candidate, current_fit, before_fit = max(validated, key=lambda row: row[0])
        if index >= len(self.maps):
            # Keep only a refinement that survives the held optical blocks from
            # both histories after adding the latest training blocks.
            prior = previous[int(np.argmax(before_fit))][0]
            combined = fit_transform(self.source, [prior, obs], candidate.transform, candidate.kind)
            if combined is not None:
                refined = combined.errors(obs)
                on_prior = combined.errors(prior)
                old_prior = active.errors(prior)
                if refined[1] <= current_fit[1] and improvement(old_prior, on_prior) >= .022:
                    candidate, current_fit = combined, refined
            self.maps.append(candidate)
        self.support = []
        self.active = index
        return dict(update, active=index, bank=len(self.maps), reason='switched', candidate=candidate.kind,
                    selected_errors=current_fit)
