import hashlib
import json
import pytest

from fly_connectome.data import Source, verify_source, write_manifest, resolve_signs, download_sources


def test_checksum_verification_detects_changed_source(tmp_path):
    path = tmp_path / 'contacts.feather'
    path.write_bytes(b'measured contacts')
    source = Source('contacts.feather', 'https://example.org/release',
                    hashlib.sha256(b'measured contacts').hexdigest())
    assert verify_source(path, source) == source.sha256
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        verify_source(path, source)


def test_manifest_records_actual_ids_sources_and_rules(tmp_path):
    path = tmp_path / 'manifest.json'
    source = Source('a', 'https://example.org/a', 'a'*64)
    write_manifest(path, [source], [10, 20], {'max_hops': 4}, 'b'*64)
    saved = json.loads(path.read_text())
    assert saved['dataset'] == 'male-cns:v1.0'
    assert saved['body_ids'] == [10, 20]
    assert saved['sources'][0]['sha256'] == 'a'*64
    assert saved['extraction_rules'] == {'max_hops': 4}


def test_unresolved_transmitters_require_explicit_policy():
    rows = [{'body': 10, 'predicted_nt': 'acetylcholine', 'predicted_nt_confidence': .9},
            {'body': 20, 'predicted_nt': 'gaba', 'predicted_nt_confidence': .8}]
    assert resolve_signs([10, 20], rows, {'acetylcholine': 1, 'gaba': -1}, .7) == [1, -1]
    with pytest.raises(ValueError, match='20'):
        resolve_signs([10, 20], rows, {'acetylcholine': 1, 'gaba': -1}, .85)
    with pytest.raises(ValueError, match='30'):
        resolve_signs([30], rows, {'acetylcholine': 1}, .7)


def test_download_verifies_before_replacing_destination(tmp_path):
    source = tmp_path/'source.bin'
    source.write_bytes(b'official data')
    manifest = tmp_path/'sources.json'
    manifest.write_text(json.dumps({'sources': [dict(filename='raw.bin', url=source.as_uri(),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest())]}))
    download_sources(manifest, tmp_path/'raw')
    assert (tmp_path/'raw/raw.bin').read_bytes() == b'official data'
    (tmp_path/'raw/raw.bin').write_bytes(b'altered')
    with pytest.raises(ValueError, match='checksum'):
        download_sources(manifest, tmp_path/'raw')
