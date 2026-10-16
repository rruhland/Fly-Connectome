"""Generic temporally correlated epsilon exploration, independent of task labels."""
import numpy as np


class BlockExploration:
    def __init__(self, frames):
        if type(frames) is not int or frames < 1:
            raise ValueError('positive integer exploration block length required')
        self.frames=frames
        self.blocks=self.samples=0
        self.reset_episode()

    def reset_episode(self):
        self.remaining=0
        self.exploratory=False
        self.action=None
        self.last_selection=None

    def choose(self,q,epsilon,rng):
        if self.remaining==0:
            self.blocks+=1
            self.exploratory=bool(rng.random()<epsilon)
            self.action=int(rng.integers(len(q))) if self.exploratory else None
            self.remaining=self.frames
        action=self.action if self.exploratory else int(rng.choice(np.flatnonzero(q==q.max())))
        self.remaining-=1
        self.samples+=1
        self.last_selection=dict(exploratory=self.exploratory,non_greedy=bool(q[action]<q.max()),
            action=action,block=self.blocks,block_frame=self.frames-self.remaining,
            block_remaining=self.remaining,sample=self.samples)
        return action
