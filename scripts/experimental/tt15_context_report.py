"""Summarize physical task scores and online decisions after each hidden change."""
import argparse
from collections import defaultdict
from pathlib import Path

from .tt4_transfer_trial import _read, _write


def summarize(receipts):
    totals = {}
    for receipt in receipts:
        site = receipt['site']
        for row in receipt['rows']:
            for mode, case in row['cases'].items():
                start = 0
                for i, (block, outcome) in enumerate(zip(row['blocks'], case['blocks'], strict=True)):
                    end = start + block['length']
                    if i:
                        key = (site,mode,block['kind'])
                        acc = totals.setdefault(key, defaultdict(int))
                        acc['trials'] += 1
                        acc['frames'] += outcome['frames']
                        acc['length'] += block['length']
                        acc['last'] += outcome['last']
                        acc['acquired'] += int(outcome['acquired'] is not None)
                        events = [event for event in case['events'] if start <= event['tick'] < end]
                        for channel in ('motor','reward'):
                            flags = [e for e in events if e[channel]]
                            acc[channel+'_blocks'] += bool(flags)
                            acc[channel+'_events'] += len(flags)
                            acc[channel+'_early'] += any(e['tick'] < start+48 for e in flags)
                        before_end = [e['selected'] for e in case['events'] if e['tick'] < end]
                        acc['correct_at_end'] += (before_end[-1] if before_end else 0) == block['motor']
                    start = end
    return [dict(site=site,mode=mode,kind=kind,**stats) for (site,mode,kind),stats in sorted(totals.items())]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipts',type=Path,nargs='+')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    rows=summarize([_read(path) for path in a.receipts])
    if a.output:
        _write(a.output,dict(rows=rows))
    for r in rows:
        print(r['site'],r['mode'],r['kind'], r['frames'],r['acquired'],r['motor_blocks'],r['reward_blocks'],r['correct_at_end'])
