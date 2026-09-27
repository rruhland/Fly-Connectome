"""Build a checksum-verified review bundle without changing production defaults."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'scripts'
    pending, modules = ['probabilistic_visual_state'], set()
    while pending:
        name = pending.pop()
        if name in modules:
            continue
        modules.add(name)
        for node in ast.walk(ast.parse((source/f'{name}.py').read_text())):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            pending.extend(n for n in imports if n and (source/f'{n}.py').exists() and n not in modules)
    files = {f'{name}.py': (source/f'{name}.py').read_bytes() for name in sorted(modules)}
    files['candidate.pt'] = (ROOT/'checkpoints/m1a5/probabilistic-visual-candidate.pt').read_bytes()
    files['README.md'] = (ROOT/'docs/plans/2026-09-26-m1a5-production-review.md').read_bytes()
    for name in ('spatial-interval-calibration', 'probabilistic-checkpoint', 'probabilistic-scenes',
                 'statistical-context'):
        path = ROOT/f'docs/experiments/2026-09-26-{name}-results.json'
        files[f'evidence/{path.name}'] = path.read_bytes()
    files['smoke.py'] = b'''from pathlib import Path
import torch
import probabilistic_visual_state

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
assert Path(probabilistic_visual_state.__file__).resolve().parent == root
model = probabilistic_visual_state.ProbabilisticVisualState.load(root/'candidate.pt')
result = model.step(torch.zeros(2, 64, 64), torch.zeros(64, 64))
assert result['support_field'].shape == (64, 64)
assert not result['entities']
assert all(len(m.keys) for m in model.dynamics.values())
print('Isolated bundled runtime and checkpoint: PASS')
'''
    revision = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                        'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(format=1, source_revision=revision, production_promoted=False,
        requirements='Python >=3.11; installed Fly-Connectome repository and its PyTorch dependency',
        validation_command='python smoke.py',
        files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    files['manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    target = ROOT/'artifacts/m1a5/probabilistic-candidate-v1.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with tempfile.TemporaryDirectory(prefix='fly-m1a5-review-') as directory:
        with zipfile.ZipFile(target) as archive:
            archive.extractall(directory)
        for name, expected in manifest['files'].items():
            assert hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest() == expected
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(ROOT/'src')
        subprocess.run([sys.executable, 'smoke.py'], cwd=directory, env=environment, check=True)
    print(json.dumps(dict(bundle=str(target), bytes=target.stat().st_size,
                          sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                          runtime_modules=len(modules), isolated_smoke=True), indent=2))


if __name__ == '__main__':
    main()
