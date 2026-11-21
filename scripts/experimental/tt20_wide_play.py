"""Replay the opening eight held points from a saved wrist game with the compact ultrawide lens."""
from .tt4_transfer_trial import _read, _write, _hash
from .tt20_camera_carry import SETTINGS, OUTPUT, Station, summarize, verify


def run():
    source = OUTPUT/'camera-carry-wrist-game-40582930-1.json.gz'
    game = _read(source)
    settings = dict(SETTINGS,wrist_optics='ultrawide',width=960,height=720)
    station = Station('wrist',settings=settings)
    rows = []
    original = game['points'][:8]
    try:
        for row in original:
            rows.append(station.point(row['seed'],row['role'],game['candidate']))
    finally:
        station.close()
    receipt = dict(schema='tt20-wrist-optics-play-1',eye='wrist',arena=settings,source_sha256=_hash(source),
                   points=rows,primary=summarize(original),wide=summarize(rows))
    def early_ball(points):
        return [[round(t) for t,_ in row['optical'] if t<=12] for row in points if row['role']==1]
    receipt['own_serve_first_13_frames'] = dict(primary=early_ball(original),wide=early_ball(rows))
    receipt['replays']=verify(receipt)
    _write(OUTPUT/'ultrawide-wrist-play.json.gz',receipt)
    print(receipt['primary'],receipt['wide'],receipt['own_serve_first_13_frames'])


if __name__ == '__main__':
    run()
