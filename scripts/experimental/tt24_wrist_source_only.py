"""Use wrist-only body recalibration with the carried reference and motor head."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import Native, ROOT
from .tt24_posture import WidePosture
from .tt24_public_body import FUSED_OUTPUT, FUSED_RECEIPT, load_public
from .tt24_public_morph import FIT_RECEIPT as PAIRED_FIT, OUTPUT as PAIRED_MAP, load_morph
from .tt24_ready import episode
from .tt24_wrist_egomotion import FIT, HEIGHT, MAP, WIDTH
from .tt24_wrist_egomotion_trial import TransferDecoder, choose_view


OUT = ROOT / 'wrist-source-only.json.gz'
ASSESSMENT = 268322197
STOCK_RETURN = 268322198


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class Calibrations:
    """Share the carried source model across all wrist-only overlays."""

    def __init__(self):
        self.prior = load_public(FUSED_OUTPUT).decoder
        with np.load(MAP, allow_pickle=False) as data:
            self.models = {key: data[key] for key in ('source_theta', 'mount', 'mount_links')}
            self.landmarks = data['points']

    def sensor(self, mode):
        if mode == 'paired_camera':
            return load_morph(PAIRED_MAP)
        eye = load_public(FUSED_OUTPUT)
        if mode in ('mount', 'mount_links'):
            eye.decoder = TransferDecoder(eye.decoder, self.prior, self.models['source_theta'], self.models[mode])
        return eye

    def recalled(self, obs):
        return choose_view(obs, self.prior, self.landmarks, self.models)


def summarize(rows):
    streaks = np.asarray([row['run'] for row in rows])
    return dict(counts=streaks.tolist(), mean=float(streaks.mean()), total=int(streaks.sum()),
                maximum=int(streaks.max()), at_least_five=int(np.sum(streaks >= 5)),
                at_least_ten=int(np.sum(streaks >= 10)))


def differences(rows, control):
    effects = np.asarray([a['run'] - b['run'] for a, b in zip(rows, control, strict=True)])
    rng = np.random.default_rng(ASSESSMENT + 2)
    combinations = rng.integers(0, len(effects), size=(10000, len(effects)))
    return dict(mean=float(effects.mean()), wins=int(np.sum(effects > 0)), ties=int(np.sum(effects == 0)),
                losses=int(np.sum(effects < 0)), interval=np.percentile(effects[combinations].mean(axis=1),
                                                                       (2.5, 97.5)).tolist())


def testing(native, weights, anchor, reuse, maps, starts, *, arm, regime, modes):
    simulator = WidePosture(view='wrist', width=WIDTH, height=HEIGHT, max_steps=360, arm=arm)
    launcher = BounceArena(view='none', arm=arm)
    try:
        for mode in modes:
            rows = []
            for start in starts:
                chosen = {}

                def sensor(obs):
                    calibrated = mode
                    if mode == 'recall':
                        calibrated, errors = maps.recalled(obs)
                        chosen.update(calibration=calibrated, reprojection=errors)
                    return maps.sensor(calibrated)

                result = episode(simulator, launcher, start, native, weights, anchor, 'scalar', regime,
                                 alpha=reuse, sensor_factory=sensor)
                rows.append(result | chosen)
                print('wrist-source-only', arm, mode, start, result['run'], chosen.get('calibration'), flush=True)
            yield mode, rows
    finally:
        simulator.close()
        launcher.close()


def earlier_starts():
    known = []
    for name in ('ready-body-combined.json.gz', 'wrist-egomotion-trial.json.gz',
                 'public-morph.json.gz', 'reference-transfer-fixed_changed.json.gz'):
        item = _read(ROOT / name)
        if isinstance(item, list):
            known.extend(row['seed'] for row in item)
        else:
            for key in ('seeds', 'alternate_seeds', 'stock_seeds', 'held_seeds'):
                known.extend(item.get(key, ()))
    return set(known)


def run(out):
    with np.load(BUNDLE, allow_pickle=False) as actor:
        weights, anchor, reuse = actor['weights'], actor['equilibrium'], float(actor['reuse'])
    wrist_exposure = _read(FIT)
    paired_exposure = _read(PAIRED_FIT)
    changed = np.random.default_rng(ASSESSMENT).integers(1, 2**31 - 1, 16).tolist()
    return_to_stock = np.random.default_rng(STOCK_RETURN).integers(1, 2**31 - 1, 4).tolist()
    if (set(changed) | set(return_to_stock)) & earlier_starts():
        raise ValueError('qualification includes an earlier held start')
    registered = dict(schema='tt24-wrist-source-only-1', held_seeds=changed, stock_seeds=return_to_stock,
                      modes=['stock', 'mount', 'mount_links', 'paired_camera'],
                      referenced_equilibrium=anchor.tolist(), reference_reuse=reuse,
                      checkpoint_hashes={p.name: digest(p) for p in (BUNDLE, FUSED_OUTPUT, MAP, PAIRED_MAP)},
                      wrist_only_calibration=dict(target_control_ticks=wrist_exposure['target_control_ticks'],
                                                  target_rgb_frames=wrist_exposure['target_rgb_frames'],
                                                  source_rgb_frames=wrist_exposure['source_rgb_frames'],
                                                  fit_clips=wrist_exposure['fit_clips'],
                                                  held_reprojection_clip=wrist_exposure['withheld_clip']),
                      paired_camera_calibration=dict(probes=len(paired_exposure['data']),
                                                      control_ticks=sum(row['ticks'] for row in paired_exposure['data'])),
                      source_camera_calibration=str(FUSED_RECEIPT),
                      new_scored_practice=0, new_optical_probes=0,
                      visual_input='on-board wrist RGB with five joint readings', held={}, summary={}, stock_return={})
    native = Native()
    try:
        maps = Calibrations()
        for mode, rows in testing(native, weights, anchor, reuse, maps, changed, arm='alternate', regime='changed',
                                  modes=registered['modes']):
            registered['held'][mode] = rows
            registered['summary'][mode] = summarize(rows)
            print('wrist-source-only summary', mode, registered['summary'][mode], flush=True)
            _write(out, registered)
        for mode, rows in testing(native, weights, anchor, reuse, maps, return_to_stock, arm='stock', regime='standard',
                                  modes=('stock', 'recall')):
            registered['stock_return'][mode] = rows
        frozen, remembered = (registered['stock_return'][name] for name in ('stock', 'recall'))
        registered['stock_exact_tape_recall'] = [a['actions'] == b['actions'] for a, b in zip(frozen, remembered, strict=True)]
        registered['paired_gains_vs_stock_map'] = {name: differences(rows, registered['held']['stock'])
                                                   for name, rows in registered['held'].items() if name != 'stock'}
        registered['physical_replays'] = sum(len(rows) for rows in (*registered['held'].values(), frozen, remembered))
        _write(out, registered)
    finally:
        native.working.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args().output)


if __name__ == '__main__':
    main()
