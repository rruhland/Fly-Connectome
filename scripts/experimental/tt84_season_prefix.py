"""Seal an equal first-block comparison while the unfiltered season continues."""

from collections import Counter
import argparse
import gzip
from hashlib import sha256
import json

from . import tt84_installation_play as play
from .tt4_transfer_trial import _write
from .tt19_view_full_game import digest_json
from .tt39_opportunity_map import projected_height


COUNT=32
ROOT=play.ROOT


def raw(path):
    p=path.read_bytes()
    return json.loads(gzip.decompress(p)),sha256(p).hexdigest()


def categories(rows):
    counts=Counter()
    for r in rows:
        for o in r['placements']['delivered']:
            if not o.get('near_bounce'):continue
            f=o.get('receive_line')
            place='missing_line' if not f else 'low' if projected_height(f)<1.01 else 'standard'
            counts[place+'_offered']+=1
            counts[place+'_landed']+=bool(o['near_landed'])
        n=r['score']['learner']; count=len(r['placements']['delivered'])
        terminal=r['placements']['delivered'][-1] if count else None
        if n==count:
            counts['no_next_legal_offer']+=1
        elif terminal and terminal['near_landed'] is False:
            counts['last_near_touch' if terminal.get('near_contact') else 'last_near_no_touch']+=1
    return dict(counts)


def run(count=COUNT):
    mname=ROOT/'season-live.json.gz'; cname=play.TT80.with_name('live.json.gz')
    m,mh=raw(mname);c,ch=raw(cname)
    if count<1 or any(len(d['rows'])<count for d in (m,c)) or m['cards'][:count]!=c['cards'][:count]:
        raise ValueError('Need the same complete first-block roster')
    a=m['rows'][:count]; b=c['rows'][:count]; pairs=[]
    for m0,c0,card in zip(a,b,m['cards'][:count],strict=True):
        pa=m0['placements']['delivered']; pb=c0['placements']['delivered']
        sa=m0['score']['learner'];sb=c0['score']['learner']
        exact=bool(pa and pb and all(digest_json(pa[0][k])==digest_json(pb[0][k]) for k in ('far_contact','near_bounce')))
        first=None if not exact else [int(pa[0]['near_landed']),int(pb[0]['near_landed'])]
        pairs.append(dict(id=card['id'],mounted=sa,stock=sb,matched_first=exact,first_own=first))
    payload=dict(n=count,snapshot_mounted_sha256=mh,snapshot_stock_sha256=ch,pairs=pairs,
                 exact_first_opportunities=sum(p['matched_first'] for p in pairs),groups={})
    for name,rows in (('mounted',a),('centered',b)):
        n=[r['score']['learner'] for r in rows]
        payload['groups'][name]=dict(total=sum(n),top=max(n),zero=n.count(0),scores=n,delivery=categories(rows))
    # Strip the live stream's current checkpoint, which belongs to a later
    # game and cannot be paired with this archived scoring prefix.
    prefix=ROOT/f'season-prefix{count}'
    if prefix.with_suffix('.json.gz').exists() or (ROOT/f'season-prefix{count}-summary.json').exists():
        raise ValueError('An immutable paired block already exists')
    _write(prefix.with_suffix('.json.gz'),dict(cards=m['cards'][:count],protocol=m['protocol'],body=m['body'],
           cap=m['cap'],mode=m['mode'],fixture_receipt=m['fixture_receipt'],rows=a,from_stream_sha256=mh))
    _write(ROOT/f'season-prefix{count}-summary.json',payload)
    print({k:v for k,v in payload.items() if k!='pairs'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--count',type=int,default=COUNT);run(p.parse_args().count)
