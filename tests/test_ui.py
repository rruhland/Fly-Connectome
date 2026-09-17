import json
import threading
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import pytest

from test_training import trainer
from fly_connectome.ui.server import make_server


def test_local_ui_serves_preview_and_queues_commands(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    trainer().save(checkpoint)
    server = make_server(checkpoint, tmp_path / 'run', port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f'http://127.0.0.1:{server.server_port}'
    try:
        assert b'Fly-Connectome' in urlopen(root).read()
        request = Request(root + '/command', data=b'{"action":"pause"}',
                          headers={'Content-Type': 'application/json'}, method='POST')
        assert json.load(urlopen(request))['queued']
        assert len(list((tmp_path / 'run').glob('command-*.json'))) == 1
        request.add_header('Origin', 'https://unrelated.example')
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


def test_evaluation_lists_preserved_bundles_and_rejects_path_escape(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    trainer().save(checkpoint)
    directory = tmp_path / 'run'
    bundles = directory / 'diagnostics'
    bundles.mkdir(parents=True)
    (bundles / 'probe-abc.json').write_text('{"seed":100}')
    server = make_server(checkpoint, directory, port=0, evaluation=True, comparison_checkpoints=[checkpoint])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urlopen(root + '/diagnostics')) == ['probe-abc']
        assert json.load(urlopen(root + '/diagnostic?id=probe-abc')) == {'seed': 100}
        assert len(json.load(urlopen(root + '/config'))['checkpoints']) == 2
        with pytest.raises(HTTPError) as error:
            urlopen(root + '/diagnostic?id=../model')
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
