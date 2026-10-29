"""Stage unique saved camera transitions for a learned visual dynamics model.

No simulator, policy fitting, role assignment, or future-action history. Source
seed identifiers are provenance/grouping only and never part of state features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pong_kernel_transition_graph import reconstruct_timing, scheduled_ticks
from pong_smdp_adaptive_forest import state_features

ROOT = Path('/workspace/m1b-experiments')
DEST = ROOT / 'pong-delta-dynamics-data-20261029-r1'
GROUPS = ROOT / 'pong-smdp-feedback2-refit-20261029-r1/data-manifest.json'
SCALES = ROOT / 'pong-smdp-closed-loop-refit-20261029-r1/smdp-data.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_scenes(path):
    with np.load(path, allow_pickle=False) as z:
        return [z['entities'][a:b].copy() for a, b in zip(z['indptr'][:-1], z['indptr'][1:])]


def sorted_rows(scene):
    scene = np.asarray(scene, np.float64).reshape(-1, 6)
    if not np.isfinite(scene).all() or not np.isin(scene[:, 4:], (0, 1)).all():
        raise ValueError('finite six-field anonymous visual observations required')
    return scene[np.lexsort(scene[:, [5, 4, 3, 2, 0, 1]].T)]


def encode(scene, phase, past_hit, scales):
    """Empty geometry is observed; overflow/missing geometry remains masked."""
    scene = sorted_rows(scene)
    geometry_valid = len(scene) <= 4
    tail_valid = 0 < len(scene) <= 4
    if tail_valid:
        return state_features(scene, phase, past_hit, scales), geometry_valid, tail_valid
    out = np.zeros(34, np.float32)
    out[28] = bool(past_hit)
    out[29 + int(phase)] = 1
    return out, geometry_valid, tail_valid


def action_histories(actions, prefix=()):
    """Oldest to newest, zero rows before true reset, never future commands."""
    actions, prefix = np.asarray(actions, np.int8), np.asarray(prefix, np.int8)
    if not np.isin(actions, (0, 1, 2)).all() or not np.isin(prefix, (0, 1, 2)).all():
        raise ValueError('literal actual primitive commands required')
    all_actions = np.r_[prefix, actions]
    history = np.zeros((len(actions), 16, 3), np.float32)
    eye = np.eye(3, dtype=np.float32)
    for i in range(len(actions)):
        end = len(prefix) + i
        selected = all_actions[max(0, end - 16):end]
        if len(selected):
            history[i, -len(selected):] = eye[selected]
    return history


def transition_trace(family, receipt_path, data_path, scene_path, *, seed,
                     ordinal=0, past_hit=False, prefix=(), extra_files=()):
    with np.load(data_path, allow_pickle=False) as z:
        data = {key:z[key].copy() for key in z.files}
    return dict(family=family, name=data_path.stem, seed=int(seed), ordinal=ordinal,
        past_hit=past_hit, prefix=np.asarray(prefix, np.int8), scenes=read_scenes(scene_path),
        actions=data['action'], hits=data['hits'].astype(bool), natural=data['terminated'].astype(bool),
        truncated=data['truncated'].astype(bool), elapsed=data['elapsed_seconds'], ticks=data['physics_ticks'],
        files=[receipt_path, data_path, scene_path, *extra_files])


def traces():
    folder = ROOT / 'pong-rally-continuous-actor-20261028-r1/training'
    for p in sorted(folder.glob('*-scores.npz')):
        stem = p.name.removesuffix('-scores.npz'); receipt_path = p.with_name(stem + '.json')
        r = json.loads(receipt_path.read_text()); scene_path = p.with_name(stem + '-scenes.npz')
        with np.load(p, allow_pickle=False) as z:
            actions, hits = z['actions'].copy(), z['hits'].astype(bool)
        n = len(actions); elapsed, ticks = reconstruct_timing(n, r['report']['physics_seconds'])
        natural = np.zeros(n, bool); truncated = natural.copy()
        if r['complete']:
            last = r['report']['rallies'][-1]
            natural[-1] = last['terminated'] and not last['truncated']
            truncated[-1] = last['truncated']
        yield dict(family='continuous60', name=stem, seed=int(r['seed']), ordinal=0, past_hit=False,
            prefix=np.array([], np.int8), scenes=read_scenes(scene_path), actions=actions, hits=hits,
            natural=natural, truncated=truncated, elapsed=elapsed, ticks=ticks,
            files=[p, receipt_path, scene_path])
    folder = ROOT / 'pong-paired-option-forest-20261029-r1/acquisition'
    for p in sorted(folder.glob('site_??.json')):
        site = json.loads(p.read_text())['site']; prefix_path = p.with_name(p.stem + '-prefix-transitions.npz')
        with np.load(prefix_path, allow_pickle=False) as z:
            prefix = z['action'].copy()
        if len(prefix) != site['anchor']:
            raise ValueError('saved actual prefix must reach declared public anchor')
        for option in range(4):
            stem = p.with_name(p.stem + f'-option_{option}')
            data = Path(str(stem) + '-transitions.npz')
            if data.exists():
                yield transition_trace('paired192suffixes', p, data, Path(str(stem) + '-scenes.npz'),
                    seed=site['seed'], ordinal=site['anchor'], past_hit=True, prefix=prefix,
                    extra_files=[prefix_path])
    for number in (1, 2):
        folder = ROOT / f'pong-smdp-onpolicy-collection-20261029-r{number}/traces'
        for p in sorted(folder.glob('site_??.json')):
            r = json.loads(p.read_text())
            if not r['transitions']:
                continue
            yield transition_trace(f'collection{number}', p, p.with_name(p.stem + '-transitions.npz'),
                p.with_name(p.stem + '-raw-scenes.npz'), seed=r['seed'])
    folders = [('c0initial8', 'pong-smdp-native-20261029-r1/screen8'),
        ('comparison1new', 'pong-smdp-cycle-comparison-20261029-r1/c1-screen8'),
        ('comparison1new', 'pong-smdp-cycle-comparison-20261029-r1/matched32/c0'),
        ('comparison1new', 'pong-smdp-cycle-comparison-20261029-r1/matched32/c1'),
        ('c2initial8', 'pong-smdp-cycle2-comparison-20261029-r1/screen8')]
    for family, folder in folders:
        for p in sorted((ROOT / folder).glob('candidate_??.json')):
            r = json.loads(p.read_text()); records = r['records']; scene_path = Path(r['structured_feature_file'])
            if not r['complete'] or len(records) != r['transitions']:
                raise ValueError('admitted native comparison must be a complete physical trace')
            yield dict(family=family, name=str(p.relative_to(ROOT)), seed=int(r['seed']), ordinal=0,
                past_hit=False, prefix=np.array([], np.int8), scenes=read_scenes(scene_path),
                actions=np.array([v['action'] for v in records], np.int8),
                hits=np.array([v['hits'] for v in records], bool),
                natural=np.array([v['terminated'] for v in records], bool),
                truncated=np.array([v['truncated'] for v in records], bool),
                elapsed=np.array([v['elapsed_seconds'] for v in records]),
                ticks=np.array([v['physics_ticks'] for v in records]), files=[p, scene_path])


def extract(output):
    began = time.monotonic()
    if output.exists():
        raise FileExistsError('do not overwrite a staged dataset')
    output.mkdir(parents=True)
    with np.load(SCALES, allow_pickle=False) as z:
        scales = z['scales'].copy()
    seed_map = {int(seed):int(start) for start, seed in json.loads(GROUPS.read_text())['start_id_to_seed'].items()}
    chunks = {}; manifest = []; families = {}; short_rows = []; known_files = {}
    for trace_index, t in enumerate(traces()):
        n = len(t['actions']); scenes = t['scenes']; ordinals = t['ordinal'] + np.arange(n)
        if len(scenes) not in (n, n + 1) or any(len(t[k]) != n for k in ('hits', 'natural', 'truncated', 'elapsed', 'ticks')):
            raise ValueError('actual camera arrays must align exactly')
        if not np.allclose(t['elapsed'], t['ticks'] / 120, rtol=0, atol=1e-14):
            raise ValueError('actual dt/ticks mismatch')
        planned_ticks = scheduled_ticks(ordinals); full = t['ticks'] == planned_ticks
        if np.any(t['ticks'] > planned_ticks) or np.any(t['ticks'] <= 0):
            raise ValueError('public scheduled camera interval mismatch')
        short = ~full
        if short.any():
            # Actual native game may stop a camera early on hit/end. Do not
            # silently relabel a short censored transition as a full outcome.
            for i in np.flatnonzero(short):
                short_rows.append(dict(trace=trace_index, row=int(i), hit=bool(t['hits'][i]),
                    natural=bool(t['natural'][i]), truncated=bool(t['truncated'][i]),
                    last=bool(i == n - 1), ticks=int(t['ticks'][i]), planned=int(planned_ticks[i])))
        past = np.r_[t['past_hit'], np.logical_or.accumulate(t['hits']) | t['past_hit']]
        encoded = [encode(s, (t['ordinal'] + i) % 5, past[i], scales) for i,s in enumerate(scenes)]
        state = np.array([e[0] for e in encoded[:n]], np.float32)
        next_state = np.zeros_like(state); available = np.arange(n) + 1 < len(scenes)
        next_state[available] = np.array([e[0] for e in encoded[1:]], np.float32)
        valid = np.array([e[1] for e in encoded[:n]], bool); tail = np.array([e[2] for e in encoded[:n]], bool)
        nv = np.zeros(n, bool); nt = nv.copy()
        nv[available] = [e[1] for e in encoded[1:]]; nt[available] = [e[2] for e in encoded[1:]]
        censor = t['truncated'].copy()
        if not t['natural'][-1] and not t['hits'][-1]:
            censor[-1] = True
        row = dict(state=state, next_state=next_state, has_next=available,
            geometry_valid=valid, tail_valid=tail, next_geometry_valid=nv, next_tail_valid=nt,
            action=np.asarray(t['actions'], np.int8), history16=action_histories(t['actions'], t['prefix']),
            history_valid=np.ones(n, bool), ordinal=ordinals.astype(np.int32),
            planned_dt=planned_ticks / 120, actualdt=np.asarray(t['elapsed'], np.float64),
            physics_ticks=np.asarray(t['ticks'], np.int8), full_interval=full,
            hits=np.asarray(t['hits'], bool), natural=np.asarray(t['natural'], bool),
            truncated=np.asarray(t['truncated'], bool), censor=censor,
            start=np.full(n, seed_map[t['seed']], np.int32), trace=np.full(n, trace_index, np.int32))
        for key, value in row.items():
            chunks.setdefault(key, []).append(value)
        stats = dict(traces=1, transitions=n, with_successor=int(available.sum()), hits=int(t['hits'].sum()),
            natural_ends=int(t['natural'].sum()), short_intervals=int(short.sum()),
            cardinality_changes=0, same_cardinality_flag_changes=0, stable_masks=0)
        for i in np.flatnonzero(available):
            a, b = sorted_rows(scenes[i]), sorted_rows(scenes[i + 1])
            if len(a) != len(b):
                stats['cardinality_changes'] += 1
            elif not np.array_equal(a[:, 4:], b[:, 4:]):
                stats['same_cardinality_flag_changes'] += 1
            else:
                stats['stable_masks'] += 1
        group = families.setdefault(t['family'], {k:0 for k in stats})
        for key, value in stats.items():
            group[key] += value
        files = []
        for p in t['files']:
            key = str(p)
            known_files.setdefault(key, sha(p))
            files.append(key)
        manifest.append(dict(index=trace_index, family=t['family'], name=t['name'], seed=t['seed'],
            start_id=seed_map[t['seed']], ordinal_start=int(t['ordinal']), past_hit_start=bool(t['past_hit']),
            actual_prefix_actions=len(t['prefix']), rows=n, files=files))
    data = {key:np.concatenate(value) for key,value in chunks.items()}
    data['scales'] = scales
    if len(manifest) != 452 or len(data['action']) != 114229 or len(np.unique(data['start'])) != 206:
        raise ValueError('unique-camera source inventory changed; review before staging')
    path = output / 'transitions.npz'; np.savez_compressed(path, **data)
    write(output / 'manifest.json', dict(schema='anonymous-camera-dynamics-data-v1', traces=manifest,
        files=known_files, start_id_to_seed={v:k for k,v in seed_map.items()},
        sources={str(p):sha(p) for p in (Path(__file__), Path(__file__).with_name('pong_smdp_adaptive_forest.py'),
            Path(__file__).with_name('pong_kernel_transition_graph.py'))}, grouping_source=str(GROUPS),
        grouping_sha256=sha(GROUPS), scales_source=str(SCALES), scales_source_sha256=sha(SCALES),
        encoding='float32C1state34; normalized horizontal anonymous padded4 sixfields+presence+pastownhit+publicphase; empty realzero geometry; overflowmasked',
        history='float32 oldest-to-newest16actualcommandonehots; true-resetpaddingzero; allpairedprefixesretained',
        censor='truncated OR finalstreamboundarywithoutownhit/naturalend; notanendlabel',
        no_new_fit=True, no_new_cameras=True))
    total = {key:sum(v[key] for v in families.values()) for key in next(iter(families.values()))}
    summary = dict(status='staged_no_fit_no_native', totals=total, families=families,
        distinct_starts=206, array_shapes={k:list(v.shape) for k,v in data.items()},
        invalid_source_geometry=int((~data['geometry_valid']).sum()), empty_or_overflow_source=int((~data['tail_valid']).sum()),
        full_event_rows=int((data['full_interval'] | data['hits'] | data['natural']).sum()),
        short_noevent_excluded_both_heads=int((~data['full_interval'] & ~data['hits'] & ~data['natural']).sum()),
        short_interval_details=short_rows, data_sha256=sha(path), data_bytes=path.stat().st_size,
        wall_seconds=time.monotonic() - began, source_frozen=all(sha(p)==h for p,h in known_files.items()))
    write(output / 'inventory.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('short_interval_details','array_shapes')}, sort_keys=True), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=DEST)
    extract(p.parse_args().output)
