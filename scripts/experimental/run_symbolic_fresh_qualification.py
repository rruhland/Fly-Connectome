"""One frozen a43/40069 fresh63297 comparison; no learning or model selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
from pathlib import Path
import time

from pong_symbolic_program import load_program
from run_rally_option_memory import ROOT, BANK, read, write, sha
from run_pong_symbolic_search import CameraBudget, qualifies
from stateful_pong_comparison import native_task, comparison
from run_rally_raw_fresh_screen import paired_metrics
import run_practical_pong_v2 as base

ORIGINAL = ROOT / 'pong-symbolic-search-20261029-r1'
ARCHIVE = ROOT / 'pong-symbolic-archive-comparison-20261029-r1'
DEST = ROOT / 'pong-symbolic-a43-fresh63297-20261029-r1'
ACTIVE_HASH = 'a43dc09a175c131c91330ef421e98aa2e7e899cde4549c21562d938d52cc3621'
ADAPTER = 'pong_symbolic_policy_adapter'


def prepare():
    assert not DEST.exists()
    old = read(ARCHIVE / 'spec.json'); result = read(ARCHIVE / 'summary.json')
    selected = next(c for c in old['candidates'] if c['active_hash'] == ACTIVE_HASH)
    assert next(c for c in result['candidates'] if c['active_hash'] == ACTIVE_HASH)['qualified']
    assert load_program(selected['model']).active_hash() == ACTIVE_HASH
    schedule = ORIGINAL / 'fresh/schedules.json'; seeds = read(schedule)['probe']
    assert len(seeds) == len(set(seeds)) == 32
    assert not set(seeds).intersection(e['seed'] for e in read(BANK)['bank'])
    assert sorted(p.name for p in (ORIGINAL / 'fresh').iterdir()) == ['schedules.json']
    assert old['source'] == base.source_signature()
    assert all(sha(p) == h for p, h in old['files'].items())
    assert all(sha(p) == h for p, h in old['assets'].items())
    DEST.mkdir(); (DEST / 'games').mkdir()
    files = dict(old['files']); files[str(Path(__file__).resolve())] = sha(__file__)
    assets = dict(old['assets'])
    for path in (schedule, ARCHIVE/'spec.json', ARCHIVE/'summary.json', Path(selected['model'])):
        assets[str(path)] = sha(path)
    spec = dict(kind='single-frozen-a43-fresh63297-pairs-v1', model=selected['model'],
        active_hash=ACTIVE_HASH, model_sha256=sha(selected['model']), seeds=seeds,
        fresh_tag=63297, original_unused_schedule=str(schedule), seconds=300,
        transition_camera_cap=64000, initial_cameras_counted_separately=True,
        workers=4, max_games=64, per_game_transition_bound=1000,
        order='baseline40069 then frozena43 per seed; interleaved four workers',
        physical_tick_limit=2400, camera_cadence='12/5 ticks',
        full_raw_chronological_candidate_replay=True, no_fit=True,
        no_parameter_changes=True, no_new_model_selection=True, no_automatic_promotion=True,
        no_automatic_extension=True, source=base.source_signature(), files=files, assets=assets)
    write(DEST/'spec.json', spec)
    print(dict(status='prepared_no_native', spec_sha256=sha(DEST/'spec.json'),
        source_sha256=sha(__file__), fresh_games=64), flush=True)


def run(approved):
    spec = read(DEST/'spec.json')
    assert approved == sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    assert spec['source'] == base.source_signature()
    assert all(sha(p) == h for p, h in spec['files'].items())
    assert all(sha(p) == h for p, h in spec['assets'].items())
    began = time.monotonic(); deadline = began + spec['seconds']
    write(DEST/'work-clock.json', dict(started_wall_ns=time.time_ns(), started_monotonic=began,
        deadline_monotonic=deadline, seconds=spec['seconds']))
    pending = [(i, seed, candidate) for i, seed in enumerate(spec['seeds']) for candidate in (False, True)]
    active = {}; results = {False: [], True: []}; budget = CameraBudget(64000)
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=base.initialize) as pool:
        def fill():
            while pending and len(active) < 4 and time.monotonic() < deadline-2:
                if not budget.reserve(1000): break
                index, seed, candidate = pending.pop(0)
                future = pool.submit(native_task, ADAPTER, spec['model'], str(DEST/'games'),
                    seed, index, deadline-1, candidate)
                active[future] = (seed, candidate)
        fill()
        while active:
            done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                seed, candidate = active.pop(future)
                try:
                    row = future.result(); assert row['transitions'] <= 1000
                except Exception as error:
                    row = dict(seed=seed, complete=False, transitions=0, initial_cameras=0,
                        transition_camera_upper_bound=1000, initial_camera_upper_bound=1,
                        unacknowledged_grant_charged=True, error=repr(error))
                budget.finish(1000, row); results[candidate].append(row)
            write(DEST/'progress.json', dict(completed={str(c):len(rows) for c,rows in results.items()},
                active_games=[dict(seed=s,candidate=c) for s,c in active.values()], cost=budget.state()))
            fill()
    groups = {c:{r['seed']:r for r in rows} for c,rows in results.items()}
    complete = all(len(rows)==32 and all(r['complete'] for r in rows.values()) for rows in groups.values())
    matched = [s for s in spec['seeds'] if all(s in groups[c] and groups[c][s]['complete'] for c in (False,True))]
    old = [groups[False][s] for s in matched]; new = [groups[True][s] for s in matched]
    stats = comparison(old,new) if matched else {}
    if complete: stats.update(paired=paired_metrics(old,new), passes_declared_comparison=qualifies(stats))
    summary = dict(status='complete' if complete else 'partial', complete=complete,
        active_hash=ACTIVE_HASH, fresh_tag=63297, complete_pairs=len(matched), comparison=stats,
        cost=budget.state(), wall_seconds=time.monotonic()-began,
        post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),
        source_frozen=all(sha(p)==h for p,h in spec['files'].items()),
        assets_frozen=all(sha(p)==h for p,h in spec['assets'].items()), no_fit=True,
        no_parameter_changes=True, no_new_model_selection=True, no_automatic_promotion=True)
    write(DEST/'summary.json',summary)
    print(dict(status=summary['status'], cost=summary['cost'],wall_seconds=summary['wall_seconds'],
        baseline_returns=stats.get('baseline',{}).get('own_returns'),
        candidate_returns=stats.get('candidate',{}).get('own_returns')),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--approved');args=parser.parse_args()
    prepare() if args.action=='prepare' else run(args.approved)
