"""Posthoc qualification of two frozen archived symbolic incumbents; no search."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
from pathlib import Path
import time

from pong_symbolic_program import Program
from run_rally_option_memory import ROOT, read, write, sha
from run_pong_symbolic_search import CameraBudget, qualifies
from stateful_pong_comparison import native_task, comparison
from run_rally_raw_fresh_screen import paired_metrics
import run_practical_pong_v2 as base


ORIGINAL = ROOT / 'pong-symbolic-search-20261029-r1'
DEST = ROOT / 'pong-symbolic-archive-comparison-20261029-r1'
ADAPTER = 'pong_symbolic_policy_adapter'
HASHES = ('a43dc09a175c131c91330ef421e98aa2e7e899cde4549c21562d938d52cc3621',
    'f85f6d53fcda9d939c3b208a4f95d7d3864fd88cd53b63d3ce2665d6787480f9')


def prepare():
    assert not DEST.exists()
    old = read(ORIGINAL / 'spec.json')
    assert old['source'] == base.source_signature()
    assert all(sha(p) == h for p, h in old['files'].items())
    assert all(sha(p) == h for p, h in old['assets'].items())
    bank = old['bank']; assert len(bank) == len({e['seed'] for e in bank}) == 32
    baseline = {e['seed']: read(e['receipt']) for e in bank}
    assert all(r['complete'] for r in baseline.values())
    assert sum(r['report']['hits'] for r in baseline.values()) == 83
    programs = {}
    for path in sorted((ORIGINAL / 'programs').glob('*.json')):
        payload = read(path)
        choices = [payload] if 'active_hash' in payload else [payload['parent'], *payload['challengers']]
        for choice in choices:
            if choice['active_hash'] in HASHES:
                program = Program.from_dict(choice)
                programs.setdefault(program.active_hash(), (program, path))
    assert set(programs) == set(HASHES)
    prior = {h: {} for h in HASHES}
    for path in sorted((ORIGINAL / 'fitness').glob('*.json')):
        receipt = read(path); h = receipt.get('program_hash')
        if h in prior and receipt.get('complete') and receipt.get('fitness_eligible'):
            seed = int(receipt['seed']); assert seed in baseline
            if seed in prior[h]:
                assert prior[h][seed]['own_hits'] == receipt['own_hits']
            prior[h][seed] = dict(seed=seed, own_hits=receipt['own_hits'], receipt=str(path), sha256=sha(path))
    DEST.mkdir()
    candidates = []
    assets = dict(old['assets'])
    assets[str(ORIGINAL / 'spec.json')] = sha(ORIGINAL / 'spec.json')
    assets[str(ORIGINAL / 'summary.json')] = sha(ORIGINAL / 'summary.json')
    for h, expected in zip(HASHES, ((18, 15), (16, 13))):
        program, source = programs[h]
        previous = list(prior[h].values())
        counts = (sum(r['own_hits'] for r in previous),
            sum(baseline[r['seed']]['report']['hits'] for r in previous))
        assert len(previous) == 8 and counts == expected
        model = DEST / (h[:12] + '.json'); write(model, program.to_dict())
        (DEST / h[:12]).mkdir()
        assets[str(model)] = sha(model); assets[str(source)] = sha(source)
        assets.update({r['receipt']: r['sha256'] for r in previous})
        candidates.append(dict(active_hash=h, model=str(model), model_sha256=sha(model),
            archive_source=str(source), prior_actual_distinct_seeds=8,
            prior_candidate_hits=counts[0], prior_baseline_hits=counts[1], prior_receipts=previous))
    files = dict(old['files']); files[str(Path(__file__).resolve())] = sha(__file__)
    spec = dict(kind='posthoc-two-archived-symbolic-incumbents-TRAIN32-v1',
        selection='exactly the two archived incumbents with positive actual return lift across eight observed distinct matched seeds; parent selected before qualification',
        candidates=candidates, bank=bank, seconds=300, transition_camera_cap=64000,
        initial_cameras_counted_separately=True, workers=4, max_games=64,
        per_game_transition_bound=1000, physical_tick_limit=2400, camera_cadence='12/5 ticks',
        order='interleave both candidates per seed; complete32 required for each qualification',
        raw_capture='all64 new native games; generic full chronological stateful replay audit',
        fitness_reuse='prior8/model are selection evidence only; no candidate games reused in qualification',
        qualification='strict total and later return gain; first parity; second count and later fraction not worse',
        no_fit=True, no_parameter_changes=True, no_fresh=True, no_automatic_promotion=True,
        no_automatic_extension=True, source=base.source_signature(), files=files, assets=assets)
    write(DEST / 'spec.json', spec)
    print(dict(status='prepared_no_native', spec_sha256=sha(DEST / 'spec.json'),
        candidates=[c['active_hash'] for c in candidates]), flush=True)


def dispatch(pool, candidates, bank, budget, deadline):
    pending = [(index, entry, candidate) for index, entry in enumerate(bank) for candidate in candidates]
    active = {}; results = {candidate['active_hash']: [] for candidate in candidates}
    def fill():
        while pending and len(active) < 4 and time.monotonic() < deadline - 2:
            if not budget.reserve(1000): break
            index, entry, candidate = pending.pop(0)
            directory = DEST / candidate['active_hash'][:12]
            future = pool.submit(native_task, ADAPTER, candidate['model'], str(directory),
                entry['seed'], index, deadline - 1, True)
            active[future] = (entry['seed'], candidate['active_hash'])
    fill()
    while active:
        done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
        for future in done:
            seed, active_hash = active.pop(future)
            try:
                result = future.result()
                assert result['transitions'] <= 1000
            except Exception as error:
                result = dict(seed=seed, complete=False, transitions=0, initial_cameras=0,
                    transition_camera_upper_bound=1000, initial_camera_upper_bound=1,
                    unacknowledged_grant_charged=True, error=repr(error))
            budget.finish(1000, result); results[active_hash].append(result)
        write(DEST / 'progress.json', dict(completed={h: len(rows) for h, rows in results.items()},
            active_games=[dict(seed=s, active_hash=h) for s, h in active.values()], cost=budget.state()))
        fill()
    return results


def run(approved):
    spec = read(DEST / 'spec.json')
    assert approved == sha(DEST / 'spec.json') and not (DEST / 'work-clock.json').exists()
    assert spec['source'] == base.source_signature()
    assert all(sha(p) == h for p, h in spec['files'].items())
    assert all(sha(p) == h for p, h in spec['assets'].items())
    began = time.monotonic(); deadline = began + spec['seconds']
    write(DEST / 'work-clock.json', dict(started_wall_ns=time.time_ns(),
        started_monotonic=began, deadline_monotonic=deadline, seconds=spec['seconds']))
    budget = CameraBudget(spec['transition_camera_cap']); outcomes = []
    baseline = {e['seed']: read(e['receipt']) for e in spec['bank']}
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=base.initialize) as pool:
        dispatched = dispatch(pool, spec['candidates'], spec['bank'], budget, deadline)
        for candidate in spec['candidates']:
            results = dispatched[candidate['active_hash']]
            byseed = {r['seed']: r for r in results}
            complete = len(byseed) == 32 and all(r['complete'] for r in results)
            valid = [byseed[e['seed']] for e in spec['bank'] if e['seed'] in byseed
                and byseed[e['seed']].get('complete')]
            old = [baseline[r['seed']] for r in valid]
            stats = comparison(old, valid) if valid else {}
            stats.update(active_hash=candidate['active_hash'], complete=complete,
                completed_games=len(valid), submitted_games=len(results),
                qualified=complete and qualifies(stats), posthoc_archive_qualification=True)
            if complete:
                stats['paired'] = paired_metrics(old, valid)
                stats['prior8_reproduced_exactly'] = all(byseed[r['seed']]['report']['hits'] == r['own_hits']
                    for r in candidate['prior_receipts'])
                assert stats['prior8_reproduced_exactly']
            write(DEST / candidate['active_hash'][:12] / 'summary.json', stats)
            outcomes.append(stats)
            write(DEST / 'interim.json', dict(candidates=outcomes, cost=budget.state()))
    summary = dict(status='complete' if all(r['complete'] for r in outcomes) else 'partial',
        candidates=outcomes, cost=budget.state(), wall_seconds=time.monotonic()-began,
        post_deadline_audit_cleanup_seconds=max(0., time.monotonic()-deadline),
        source_frozen=all(sha(p) == h for p, h in spec['files'].items()),
        assets_frozen=all(sha(p) == h for p, h in spec['assets'].items()),
        no_fit=True, no_parameter_changes=True, no_fresh=True, no_automatic_promotion=True)
    write(DEST / 'summary.json', summary)
    print(dict(status=summary['status'], cost=summary['cost'], wall_seconds=summary['wall_seconds'],
        candidates=[dict(active_hash=r['active_hash'], complete=r['complete'], qualified=r['qualified'],
            own_returns=r.get('candidate', {}).get('own_returns'),
            later_returns=r.get('candidate_more', {}).get('later_returns')) for r in outcomes]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--approved'); args = parser.parse_args()
    prepare() if args.action == 'prepare' else run(args.approved)
