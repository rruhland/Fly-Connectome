"""Visual-only indoor practice hall surrounding the regulation MuJoCo station."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def hall_model(path):
    doc = path if isinstance(path, ET.Element) else ET.parse(path).getroot()
    world = doc.find("worldbody")
    asset = doc.find("asset")
    ET.SubElement(asset, "texture", name="maple_boards", type="2d", builtin="flat",
                  width="512", height="512", rgb1=".61 .43 .29")
    ET.SubElement(asset, "material", name="hardwood", texture="maple_boards",
                  texrepeat=".38 .38", texuniform="true", reflectance=".05", shininess=".25")
    floor = world.find("geom[@name='floor']")
    floor.set("material", "hardwood")
    floor.set("rgba", "1 1 1 1")

    def box(name, pos, size, rgb):
        return ET.SubElement(world, "geom", name=name, type="box", pos=pos, size=size,
                             rgba=rgb + " 1", contype="0", conaffinity="0", group="1")

    for side in (-1, 1):
        box(f"end_wall_{side}", f"0 {side*5.9} 2.1", "5.8 .06 2.1", ".72 .75 .72")
        box(f"end_wall_base_{side}", f"0 {side*5.82} .59", "5.8 .025 .58", ".27 .35 .36")
        box(f"long_wall_{side}", f"{side*5.6} 0 2.1", ".07 5.9 2.1", ".68 .72 .71")
        box(f"court_line_{side}", f"{side*2.4} 0 .001", ".018 3.2 .001", ".85 .84 .74")
    for x in (-4, -2, 0, 2, 4):
        box(f"high_window_{x}", f"{x} 5.81 2.65", ".73 .025 .43", ".58 .71 .78")
        box(f"window_mullion_{x}", f"{x} 5.77 2.65", ".022 .02 .44", ".39 .46 .46")
        box(f"window_sill_{x}", f"{x} 5.76 2.19", ".77 .05 .022", ".39 .46 .46")
    for i, (x,y) in enumerate([(-2.2,2.95),(0,3.25),(2.2,2.95)]):
        box(f"court_barrier_{i}", f"{x} {y} .365", ".96 .018 .305", ".075 .20 .31")
        box(f"barrier_rail_{i}", f"{x} {y} .682", ".98 .022 .012", ".59 .64 .64")
        for j, offset in enumerate((-.83,.83)):
            box(f"barrier_foot_{i}_{j}", f"{x+offset} {y} .044", ".021 .21 .042", ".19 .22 .23")
    for i, x in enumerate((-3.2, 3.2)):
        box(f"bench_{i}", f"{x} 4.12 .43", ".78 .18 .032", ".55 .36 .19")
        for j, offset in enumerate((-.66,.66)):
            box(f"bench_leg_{i}_{j}", f"{x+offset} 4.12 .20", ".035 .15 .20", ".23 .25 .24")
    ET.SubElement(world, "light", name="hall_fill", pos="1.8 3.8 3.7", dir="-.2 -.60 -.75",
                  diffuse=".43 .44 .47", specular=".12 .12 .12", castshadow="false")

    model = mujoco.MjModel.from_xml_string(ET.tostring(doc, encoding="unicode"))
    # Repeatable grain, staggered seams and slight plank-to-plank color variation.
    tex = model.texture("maple_boards")
    w, h = int(tex.width[0]), int(tex.height[0])
    y, x = np.indices((h, w))
    plank = x // 32
    joint = (y + 87 * plank) % 256
    grain = 3 * np.sin(.9 * x + .25 * np.sin(y / 13)) + 1.3 * np.sin(.16 * x + .28 * np.sin(y / 51))
    variation = 6 * np.sin(plank * 4.2 + (y + 87 * plank) // 256)
    lightness = grain + variation - 20 * ((x % 32 == 0) | (joint == 0))
    tex.data[:] = np.clip(np.array([178, 138, 99]) + lightness[..., None], 0, 255).astype(np.uint8)
    return model


def hall_colors(rng):
    """Randomize each visible rubber face and regulation white/orange ball."""
    rubber = np.array([[43, 37, 39], [192, 46, 48]])
    ball = np.array([[238, 235, 222], [239, 147, 55]])
    faces = rubber[rng.integers(2, size=2)]
    return np.vstack((faces, ball[rng.integers(2)], [164, 168, 158])).astype(np.uint8)
