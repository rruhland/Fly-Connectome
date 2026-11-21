"""Play one recorded rebound catch twice using the original RGB camera."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .tt4_replay import recording
from .tt4_transfer_trial import _read
from .tt8_transfer_rebound import DT, HEIGHT, WIDTH, ReboundCourt, steer
from .tt8_transfer_trial import EARLY_FRAMES


PLAYERS = {'matched': 'Table-tennis practice', 'target_hard': 'Target only · hard prior'}


def annotate(images, heading, footer):
    width, height = WIDTH * 2, HEIGHT * 2
    output = Image.new('RGB', (2 * width + 6, height + 86), '#141d25')
    draw = ImageDraw.Draw(output)
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    small = ImageFont.truetype('DejaVuSans.ttf', 17)
    draw.text((12, 8), heading, fill='#f1f4f9', font=font)
    for i, (name, frame) in enumerate(images.items()):
        offset = i * (width + 6)
        draw.text((offset + 12, 40), PLAYERS[name], fill='#bce2fa', font=small)
        output.paste(Image.fromarray(frame).resize((width, height), Image.Resampling.NEAREST), (offset, 61))
        draw.text((offset + 12, height + 65), footer[name], fill='#f1f4f9', font=small)
    return output


def replay(receipt, output, fall):
    row = next(entry for entry in receipt['rows'] if entry['visibility'] == 'curtain' and entry['fall'] == fall
               and entry['players']['matched']['catch'] and not entry['players']['target_hard']['catch'])
    court = ReboundCourt(row['material'])
    opening = court.opening(row['fall'])
    carts = dict.fromkeys(PLAYERS, opening.cart_start)
    scene = f'Rebound catch · {row["material"]} plate'

    def frames(ball, label):
        return annotate({name: court.frame(ball, cart) for name, cart in carts.items()}, scene, label)

    labels = dict.fromkeys(PLAYERS, 'Ball visible')
    with recording(output, frames(opening.positions[0], labels), DT * 2.5) as append:
        count = len(opening.positions) - 1
        for i in range(count):
            step = opening.crossing_time - i * DT if i == count - 1 else DT
            for name in PLAYERS:
                carts[name] = steer(carts[name], row['players'][name]['commands'][i], step)
            if i == count - 1:
                labels = {name: 'Caught' if row['players'][name]['catch'] else 'Missed' for name in PLAYERS}
                append(frames(opening.positions[-1], labels))
            elif i + 1 >= EARLY_FRAMES:
                append(frames(None, dict.fromkeys(PLAYERS, 'Ball behind screen')))
            else:
                append(frames(opening.positions[i + 1], labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('fall', type=int)
    args = parser.parse_args()
    replay(_read(args.receipt), args.output, args.fall)
