"""Opt-in delayed competition over causal local assembly paths."""

import math


class MultiHypothesisAssemblies:
    def __init__(self, *, affinity=None, radius=3, max_gap=8,
                 beam=4, max_tracks=16):
        self.affinity = affinity
        self.radius = radius
        self.max_gap = max_gap
        self.beam = beam
        self.max_tracks = max_tracks
        self.reset_state()

    def reset_state(self):
        self.recent = []
        self.all_hypotheses = []

    def edge_score(self, hypothesis, candidate, frame):
        y, x, _ = candidate
        last_frame, last_y, last_x = hypothesis['history'][-1]
        age = frame-last_frame
        distance = max(abs(y-last_y), abs(x-last_x))
        if age < 1 or age > self.max_gap or distance > self.radius:
            return None
        score = 1-distance/(self.radius+1)-.1*(age-1)
        if len(hypothesis['history']) >= 2:
            _, older_y, older_x = hypothesis['history'][-2]
            old_dy, old_dx = last_y-older_y, last_x-older_x
            new_dy, new_dx = y-last_y, x-last_x
            old_norm = math.hypot(old_dy, old_dx)
            new_norm = math.hypot(new_dy, new_dx)
            if old_norm and new_norm:
                score += .5*(old_dy*new_dy+old_dx*new_dx)/(
                    old_norm*new_norm)
        if self.affinity is not None:
            source = (last_y, last_x, hypothesis['feature'])
            score += 2*self.affinity.score(source, candidate)
        return score

    def step(self, frame, candidates):
        predecessors = [row for group in self.recent for row in group
                        if frame-row['history'][-1][0] <= self.max_gap]
        new = []
        for y, x, feature in candidates:
            candidate = (y, x, feature)
            options = [dict(history=[(frame, y, x)],
                            feature=feature.clone(), score=0.)]
            for previous in predecessors:
                edge = self.edge_score(previous, candidate, frame)
                if edge is None:
                    continue
                options.append(dict(history=previous['history']+[(frame, y, x)],
                                    feature=(previous['feature']+feature)/2,
                                    score=previous['score']+edge))
            options.sort(key=lambda row: row['score'], reverse=True)
            new.extend(options[:self.beam])
        self.recent = [group for group in self.recent
                       if group and frame-group[0]['history'][-1][0]
                       < self.max_gap]
        self.recent.append(new)
        self.all_hypotheses.extend(new)

    def tracks(self):
        ranked = sorted(self.all_hypotheses,
                        key=lambda row: (len(row['history']), row['score']),
                        reverse=True)
        chosen = []
        used = set()
        for row in ranked:
            if len(row['history']) < 2:
                continue
            nodes = set(row['history'])
            if len(nodes & used)*2 >= len(nodes):
                continue
            chosen.append(row)
            used |= nodes
            if len(chosen) == self.max_tracks:
                break
        return chosen
