import hashlib
from fly_connectome.viewer import catalog


def test_source_pin_allows_only_windows_newline_conversion(tmp_path, monkeypatch):
    root = tmp_path / 'fly_connectome'
    (root / 'viewer').mkdir(parents=True)
    source = root / 'pong.py'
    expected = {'pong.py': hashlib.sha256(b'x = 1\ny = 2\n').hexdigest()}
    monkeypatch.setattr(catalog, '__file__', str(root / 'viewer' / 'catalog.py'))
    monkeypatch.setattr(catalog, 'source_signature', lambda: {
        'pong.py': hashlib.sha256(source.read_bytes()).hexdigest()})
    source.write_bytes(b'x = 1\r\ny = 2\r\n')
    assert catalog.runtime_source_matches(expected)
    source.write_bytes(b'x = 9\r\ny = 2\r\n')
    assert not catalog.runtime_source_matches(expected)
    assert not catalog.runtime_source_matches({})
