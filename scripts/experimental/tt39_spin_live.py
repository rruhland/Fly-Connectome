"""A brief matched live test changing only inferred spin in the optical mentor's private twin."""

import argparse
from collections import Counter
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np

from . import tt37_visual_inertia as optics
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_fast_camera import native_ball_pixels
from .tt35_error import triangulate
from .tt38_contact_id import OLD, OUT as TT38, VISION
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt39_spin import Observer, solve


OUT = Path('artifacts/tt39')
PLAN = TT38/'contact-id-plan.json'
REFERENCE = TT38/'contact-id-live-nominal.json.gz'
MEMORY = OLD/'reuse-warm-frozen.json.gz'
LIVE_MODE = 'coupled'  # the lower delayed-RGB error on optical practice, not a live choice


class AngularReadout:
    """Replace only the rotation callback, never the earlier p/v estimate."""

    def __init__(self, legacy, settings, capture_dt, counter):
        self.legacy = legacy
        self.cfg = settings
        self.readout = Observer(capture_dt, settings['settings'])
        self.counter = counter
        self.last_t = -5.
        self.fit = None

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def rotation(self, now):
        self.counter['all_syncs'] += 1
        if not self.readout.samples:
            return self.legacy.rotation(now)
        latest = self.readout.samples[-1][0]
        if now-latest > .043:
            return self.legacy.rotation(now)
        if latest != self.last_t:
            self.last_t = latest
            self.fit = None
            sample = self.readout.evidence()
            if sample is not None:
                ts, positions, packet, stamp, bounce = sample
                if bounce is not None:
                    self.fit = solve(ts, positions, packet, self.cfg['tuned'][LIVE_MODE],
                                     LIVE_MODE, stamp, bounce)
        if self.fit is None:
            return self.legacy.rotation(now)
        self.counter['inferred_syncs'] += 1
        return self.fit.state(now)[1]


class MultirateOptics:
    """Leave p/v on its exact old 30 Hz images. Only angular evidence sees extra frames."""

    def __init__(self, factory, tracker, checkpoint, settings, counter):
        self.base = factory(tracker, checkpoint)
        self.filter = AngularReadout(self.base.filter, settings, 1/60, counter)
        self.latest = None

    def observe(self, now, frames, q):
        if int(round(60*now)) % 2 == 0:
            self.latest = self.base.observe(now, frames, q)
        f = self.latest
        if f is not None and now-f.time <= .11 and f.position[1] < .65 and f.velocity[1] < -.85:
            last = self.filter.legacy.track[-1] if self.filter.legacy.track else None
            if last is not None and abs(last[0]-now) < 1e-8:
                p = last[1]
            else:
                pixels = [native_ball_pixels(frames[k]) for k in ('left','right')]
                p = (None if any(e is None or e.clipped for e in pixels)
                     else triangulate([e.center for e in pixels], self.base.tracker.rig))
            self.filter.readout.ingest(now,p)
        return self.latest


def compare(tape, control, counts):
    if tape['cards'] != control['cards'] or len(tape['rows']) != len(control['rows']):
        raise ValueError('Not a complete paired set')
    ready_delta = float(np.max(np.abs(np.asarray(tape['ready'])-control['ready'])))
    if ready_delta > 1e-12:
        raise ValueError('Nonangular live setting changed: ready')
    for key in ('memory_start','depth','cut','mode','economy','repair','robust','body','visual',
                'cap','refine','rounds','samples','guided','group'):
        if tape[key] != control[key]:
            raise ValueError(f'Nonangular live setting changed: {key}')
    rows = []
    for card,a,b in zip(tape['cards'],control['rows'],tape['rows'],strict=True):
        for r in (a,b):
            if digest_json(audit(r['events'],opening_side=1)) != digest_json(r['score']):
                raise ValueError('Scoring changed on independent audit')
        aa, bb = (next((e for e in r['events'] if e['kind']=='contact' and e['side']==-1), None)
                  for r in (a,b))
        if aa != bb:
            raise ValueError('Paired opening far contact changed')
        rows.append(dict(id=card['id'], seed=card['seed'], reply_seed=card['reply_seed'],
                         base=a['score']['learner'], spin=b['score']['learner'],
                         base_total=a['score']['returns'], spin_total=b['score']['returns']))
    return dict(role='optical full-game mentor, not direct-joint pupil', rows=rows,
                unchanged_legal=sum(r['base'] for r in rows), angular_legal=sum(r['spin'] for r in rows),
                calls=None if counts is None else dict(counts), ready_roundoff_rad=ready_delta,
                independent_scored_histories=2*len(rows))


def summarize(checkpoint, output, summary):
    observed = _read(output)
    base = _read(REFERENCE)
    n = len(observed['rows'])
    if n != len(observed['cards']):
        raise ValueError('Live angular trial did not finish all prescribed games')
    control = dict(base, cards=base['cards'][:n], rows=base['rows'][:n])
    report = compare(observed, control, observed.get('rotation_reads'))
    report.update(angular_checkpoint_sha256=_hash(checkpoint), stationary_angular_baseline_sha256=_hash(REFERENCE),
                  source_plan_sha256=_hash(PLAN), source_memory_sha256=_hash(MEMORY),
                  angular_tape_sha256=_hash(output), mode=LIVE_MODE)
    _write(summary,report)


def run(checkpoint, output, summary, n=4):
    cfg = _read(checkpoint)
    logged = Counter()
    options = ['tt37_motor_clinic','optical-fixed',str(output),'--plan',str(PLAN),
               '--group','held','--n',str(n),'--depth','-1.56','--cut','.18','--samples','35','--rounds','3',
               '--guided','--refine','--body','--cap','24','--visual',str(VISION),
               '--memory',str(MEMORY),'--camera-stride','1']
    original = optics.LearnedOptics
    def with_spin(tracker, path):
        return MultirateOptics(original, tracker, path, cfg, logged)
    with patch.object(optics, 'LearnedOptics', with_spin), patch.object(sys, 'argv', options):
        runpy.run_module("scripts.experimental.tt37_motor_clinic",run_name="__main__")
    _write(output, dict(_read(output), rotation_reads=dict(logged)))
    summarize(checkpoint, output, summary)


def verify_replays(output, summary):
    result = _read(summary)
    if result['angular_tape_sha256'] != _hash(output) or result['stationary_angular_baseline_sha256'] != _hash(REFERENCE):
        raise ValueError('One paired live tape changed')
    score = _read(output)
    prior = _read(REFERENCE)
    compared = 0
    for game in (score, prior):
        with centered(game['ready']):
            court = arena(protocol='random')
            try:
                cards, paired_rows = game['cards'][:len(score['rows'])], game['rows'][:len(score['rows'])]
                for card, row in zip(cards,paired_rows,strict=True):
                    court.launch(row['shot'],card=card)
                    court.partner.reset(seed=card['seed'])
                    for cmd in row['actions']:
                        court.exchange(cmd,FAR_COMMAND)
                    if digest_json(court.events) != digest_json(row['events']):
                        raise ValueError('Private-controller action tape diverged on physical replay')
                    compared += 1
            finally:
                court.close()
    _write(summary, dict(result, fresh_physical_replays=compared))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=OUT/'spin-checkpoint.json')
    p.add_argument('--output',type=Path,default=OUT/'spin-live.json.gz')
    p.add_argument('--summary',type=Path,default=OUT/'spin-live-summary.json')
    p.add_argument('--n',type=int,default=4)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    a = p.parse_args()
    if a.audit:
        verify_replays(a.output,a.summary)
    elif a.summarize_only:
        summarize(a.checkpoint,a.output,a.summary)
    else:
        run(a.checkpoint,a.output,a.summary,a.n)
