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
