"""Retain supported already-made preparation through ambiguous stereo updates."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt54_live_wing as first

rp=first.rp


class Wing(first.Wing):
    def reset(self):
        super().reset()
        self.last_reliable=None;self.last_visible=None;self.last_goal=None

    def look(self,tick,x,ready):
        if ready.predictor is None or ready.predictor.future is None:return
        previous=self.level
        super().look(tick,x,ready)
        if not self.logs or self.logs[-1].get('action')!='far optical evidence' or self.logs[-1]['tick']!=tick:
            return
        lane,height,risk=self.forecast
        if lane<-.045 and risk>.24 and height<1.035:
            self.last_reliable=tick
        elif self.last_reliable is not None and tick-self.last_reliable<=25 and not self.declined:
            # Later optical ambiguity is not a mechanical request to undo a
            # supported action. Keep most acquired progress while newer right
            # or clearly high evidence still vetoes exactly as before.
            self.level=max(self.level,previous-.035)
            self.logs[-1].update(level=self.level,retained_earlier_supported_evidence=True)

    def command(self,tick,x,q,ref,plain,p):
        if x is None and self.opened is not None and not self.ended and not self.declined:
            if self.last_goal is not None and self.last_visible is not None and tick-self.last_visible<=4:
                return self.follow(q,ref,self.last_goal)
        if x is not None:self.last_visible=tick
        g=super().command(tick,x,q,ref,plain,p)
        if g is not None:self.last_goal=np.asarray(g).copy()
        return g


def install(raw,checker,reference,*,enabled,frames):
    # Reuse the tested pair factory unchanged apart from early body tracking.
    p=first.install(raw,checker,reference,enabled=False,frames=frames)
    if enabled:p.wing=Wing(rp._read(first.BANK),checker)
    return p


def run(path):
    fx=first.install
    def make(*a,**kw):
        p=fx(*a,**{**kw,'enabled':False})
        if kw['enabled']:p.wing=Wing(rp._read(first.BANK),a[1])
        return p
    with patch.object(first,'install',make):
        first.run(path)
    d=rp._read(path);d['wing_extension_sha256']=rp._hash(Path(__file__));rp._write(path,d)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=first.root/'retained-exact.json.gz')
    args=p.parse_args();first.root.mkdir(parents=True,exist_ok=True);run(args.out)
