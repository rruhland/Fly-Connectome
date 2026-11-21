"""Render the actual four receiver mechanisms from the same side of the table."""
import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from .tt6_body_arena import TT6Arena


def sheet(output):
    entries = [
        ('stock', 'Original arm', '4 rotary axes'),
        ('gantry', 'Gantry', '3 slides and a wrist'),
        ('offset', 'Oblique stage', '5 active axes including a tilted pivot'),
        ('rotary', 'Articulated arm', '6 rotary axes'),
    ]
    width, height = 390, 270
    result = Image.new('RGB', (2*width, 2*(height+65)), '#eff1ed')
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype('DejaVuSans.ttf', 17)
    bold = ImageFont.truetype('DejaVuSans-Bold.ttf', 20)
    for i,(body,label,detail) in enumerate(entries):
        arena = TT6Arena(91529, view_seed=1001, body=body, render_mode='mujoco', venue='hall')
        try:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = (0., -1.55, .83)
            camera.azimuth, camera.elevation, camera.distance = 44., -16., 1.55
            with mujoco.Renderer(arena.model, height=height, width=width) as renderer:
                renderer.update_scene(arena.data, camera=camera)
                rgb = renderer.render()
            x,y = i % 2 * width, i // 2 * (height+65)
            result.paste(Image.fromarray(rgb), (x,y))
            draw.text((x+12,y+height+5), label, fill='#182729', font=bold)
            draw.text((x+12,y+height+34), detail, fill='#3c5151', font=font)
        finally:
            arena.close()
    result.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    sheet(parser.parse_args().output)
