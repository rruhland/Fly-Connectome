import ctypes
import subprocess
from pathlib import Path

import mujoco
import numpy as np


NATIVE = Path(__file__).resolve().parents[1] / 'scripts/experimental/tt29_rally_rollout.cpp'


def bind_bounds(tmp_path):
    source = tmp_path / 'clearance.cpp'
    source.write_text(f'''#include "{NATIVE}"
extern "C" bool separated(const mjModel* m, const mjData* d, int i, int j, double horizon) {{
    const auto a = local_clearance_box(m,i), b = local_clearance_box(m,j);
    return a.finite && b.finite && boxes_separated(world_clearance_box(d,i,a),world_clearance_box(d,j,b),horizon);
}}
''')
    package = Path(mujoco.__file__).parent
    lib = next(package.glob('libmujoco.so.*'))
    tool = tmp_path / 'bounds.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp',
                    '-I', str(package / 'include'), str(source), str(lib),
                    f'-Wl,-rpath,{package}', '-o', str(tool)], check=True)
    run = ctypes.CDLL(str(tool)).separated
    run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    run.restype = ctypes.c_bool
    return run


def test_conservative_world_bounds_preserve_close_and_penetrating_geometries(tmp_path):
    test = bind_bounds(tmp_path)
    xml = '''<mujoco>
      <asset>
        <mesh name="asymmetric" vertex="0 0 0 .10 0 0 .08 .07 0 0 .05 0
                                            .01 .02 .09 .11 .01 .07 .06 .08 .08 .02 .06 .08"/>
      </asset>
      <worldbody>
        <geom type="plane" size="1 1 .01"/>
        <geom type="box" size=".13 .067 .025"/>
        <geom type="cylinder" size=".063 .016"/>
        <geom type="capsule" size=".036 .089"/>
        <geom type="ellipsoid" size=".045 .059 .084"/>
        <geom type="sphere" size=".045"/>
        <geom type="mesh" mesh="asymmetric"/>
      </worldbody>
    </mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(40101)
    skipped = tight = overlap = 0
    for scale in [.065, .15, .3]:
        for _ in range(100):
            m.geom_pos[1:] = rng.normal(size=(6, 3)) * scale
            q = rng.normal(size=(6, 4))
            m.geom_quat[1:] = q / np.linalg.norm(q, axis=1, keepdims=True)
            mujoco.mj_kinematics(m, d)
            for i in range(1, m.ngeom):
                assert not test(m._address, d._address, 0, i, .004)
                for j in range(i+1, m.ngeom):
                    for limit in (.001, .004, .02):
                        sep = test(m._address, d._address, i, j, limit)
                        distance = mujoco.mj_geomDistance(m, d, i, j, limit, None)
                        if sep:
                            skipped += 1
                            assert distance >= limit - 1e-10
                        if distance < limit:
                            tight += 1
                            overlap += distance < -.003
                            assert not sep
    assert skipped > 1000 and tight > 100 and overlap > 100
