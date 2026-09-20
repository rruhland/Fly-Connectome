import json
import subprocess
import sys

from test_training import trainer


def test_headless_training_and_evaluation_commands(tmp_path):
    initial, trained = tmp_path/'initial.pt', tmp_path/'trained.pt'
    trainer().save(initial)
    command = [sys.executable, '-m', 'fly_connectome']
    result = subprocess.run(command + ['train', str(initial), '--steps', '2', '--output', str(trained), '--checkpoint-every', '1'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert trained.exists()
    progress = [json.loads(line) for line in result.stdout.splitlines()]
    assert [item['step'] for item in progress[:-1]] == [2]  # fixture synchronizes every two steps
    assert all(item['event']=='checkpoint' for item in progress[:-1])
    assert progress[-1]['completed_steps']==2
    report = tmp_path/'report.json'
    result = subprocess.run(command + ['evaluate', str(trained), '--initial', str(initial),
        '--seeds', '100', '--steps', '2', '--output', str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())['learned']['hit_shaping'] == 0
