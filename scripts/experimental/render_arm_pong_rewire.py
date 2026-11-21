"""Replay a scheduled match from physical joint commands and export both real cameras."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from arm_plays_pong import NATIVE_HZ
from arm_pong_rewire import SWITCH, run_case, schedule
from fly_connectome.pong import Physics, Pong
from pong_button_panel import ButtonPanel, SOURCE_GEAR, decode_panel
from return_metrics import ReturnCounter


def replay_physics(case, rows, *, opponent_speed=.45, switch=SWITCH):
    arm = ButtonPanel(case.panel_seed, wiring_seed=case.wiring_seed, gear=SOURCE_GEAR)
    game = Pong([case.pong_seed], config=Physics(opponent_speed=opponent_speed))
    counter = ReturnCounter()
    drives = {direction: torch.tensor([-float(direction)]) for direction in (-1, 0, 1)}
    counts = dict(ticks=0, returns=0, opportunities=0, won=0, lost=0)
    for row in rows:
        tick = row['tick']
        if tick == switch:
            arm.rewire()
        drive = arm.step(row['action'])
        if drive != row['drive'] or arm.active_button != row['active_button']:
            raise ValueError(f'physical button mismatch at {tick}')
        old = counter.opportunities
        outcome = counter.step(game, game.step, drives[drive], auto_reset=False)
        score, hit = int(outcome.scores[0]), int(outcome.hits[0])
        incoming = int(counter.opportunities > old)
        if (hit, incoming, score) != (row['hit'], row['opportunity'], row['score']):
            raise ValueError(f'native Pong mismatch at {tick}')
        counts['ticks'] += 1
        counts['returns'] += hit
        counts['opportunities'] += incoming
        counts['won'] += int(score == 1)
        counts['lost'] += int(score == -1)
        if score:
            game.reset(torch.ones(1, dtype=torch.bool))
            counter = ReturnCounter()
    return counts


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=size)
    except OSError:
        return ImageFont.load_default()


class Film:
    def __init__(self, *, switch=SWITCH):
        self.switch = switch
        self.images = []
        self.screen = None
        self.ticks = 0
        self.head = font(19)
        self.small = font(14)
        self.score = self.against = self.returns = self.chances = 0

    def observe(self, row):
        tick = row['tick']
        self.ticks = tick + 1
        if row['screen'] is not None:
            self.screen = row['screen']
        if tick % 8 == 0:
            frame = Image.new('RGB', (730, 385), '#12212b')
            draw = ImageDraw.Draw(frame)
            draw.text((14, 9), 'Pong through a three-joint arm', font=self.head, fill='#ffffff')
            draw.text((550, 11), f'{tick / NATIVE_HZ:05.2f} s    Score {self.score}–{self.against}',
                      font=self.small, fill='#f8df9e')
            court = Image.fromarray(np.where(self.screen, 255, 0).astype('uint8'))
            frame.paste(court.resize((264, 264), Image.Resampling.NEAREST), (14, 62))
            arm = Image.fromarray(row['arm'])
            frame.paste(arm.resize((264, 264), Image.Resampling.NEAREST), (292, 62))
            center = decode_panel(row['arm']).neutral.round().astype(int)
            xy = tuple(center - 13) + tuple(center + 13)
            frame.paste(arm.crop(xy).resize((145, 145), Image.Resampling.NEAREST), (570, 62))
            for x, label in [(14, 'Pong screen'), (292, 'Overhead arm camera'), (570, 'Button close-up')]:
                draw.text((x, 39), label, font=self.small, fill='#abcedb')
            for y, color, label in [(220, '#ffbc42', 'Fingertip'), (245, '#eb53b7', 'First button'),
                                     (270, '#45e363', 'Second button'), (295, '#60dded', 'Rest position')]:
                draw.ellipse((572, y + 3, 581, y + 12), fill=color)
                draw.text((590, y), label, font=self.small, fill='#f1f4ef')
            status = 'original' if self.switch is not None else 'held constant'
            if self.switch is not None and tick >= self.switch:
                status = f'reversed at {self.switch / NATIVE_HZ:g} s'
            meaning = row['inferred_down_button']
            label = 'testing' if meaning is None else ('pink = down' if meaning == 0 else 'green = down')
            draw.text((14, 336), f'Wiring  {status}',
                      font=self.small, fill='#f8df9e')
            draw.text((292, 336), f'Learned mapping  {label}', font=self.small, fill='#f1f4ef')
            draw.text((570, 336), f'Returns  {self.returns}/{self.chances}', font=self.small, fill='#f1f4ef')
            draw.text((14, 359), 'Both cameras are learner inputs. Labels and totals are for the viewer.',
                      font=self.small, fill='#abc0c9')
            self.images.append(frame)
        self.score += row['score'] == 1
        self.against += row['score'] == -1
        self.returns += row['hit']
        self.chances += row['opportunity']

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        durations = [round((i+1)*1000/15) - round(i*1000/15) for i in range(len(self.images))]
        final = self.images[-1].copy()
        draw = ImageDraw.Draw(final)
        draw.rectangle((0, 328, 730, 385), fill='#12212b')
        draw.rectangle((520, 0, 729, 37), fill='#12212b')
        draw.text((550, 11), f'{self.ticks / NATIVE_HZ:05.2f} s    Score {self.score}–{self.against}',
                  font=self.small, fill='#f8df9e')
        draw.text((14, 336), f'Final score   Arm {self.score}   Opponent {self.against}',
                  font=self.head, fill='#f8df9e')
        draw.text((392, 339), f'Returns   {self.returns} of {self.chances} incoming balls',
                  font=self.small, fill='#f1f4ef')
        draw.text((14, 363), 'Last sampled camera images, with final native game totals.',
                  font=self.small, fill='#abc0c9')
        self.images[0].save(path, save_all=True, append_images=[*self.images[1:], final],
                            duration=[*durations, 1250],
                            loop=0, lossless=True, method=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('webp', type=Path)
    parser.add_argument('tape', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    case = schedule()[0]
    film = Film()
    result = run_case(case, args.checkpoint, callback=film.observe, return_trace=True)
    reference = json.loads(args.receipt.read_text())['matches'][0]['adaptive']
    rows = result.pop('trajectory')
    for field in ('ticks', 'phases', 'points', 'contacts', 'binding_events', 'arm_actions_sha256', 'drives_sha256'):
        if json.loads(json.dumps(result[field])) != reference[field]:
            raise ValueError(f'recorded run differs from the registered match in {field}')
    verified = replay_physics(case, rows)
    args.tape.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.tape, 'wt', encoding='utf-8') as out:
        json.dump(dict(case=result['case'], switch_tick=SWITCH, replay_counts=verified, trajectory=rows),
                  out, separators=(',', ':'))
    film.save(args.webp)
    print(json.dumps(verified))


if __name__ == '__main__':
    main()
