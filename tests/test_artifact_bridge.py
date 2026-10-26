"""Loopback-only bridge integration/security tests; no real credentials/network."""
import hashlib
from http.client import HTTPConnection
import importlib.util
import io
import json
import os
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from urllib.parse import quote
import pytest

spec=importlib.util.spec_from_file_location('artifact_bridge',Path(__file__).parents[1]/'scripts/artifact_bridge.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
TOKEN='a'*64


@pytest.fixture
def receiver(tmp_path):
    root=tmp_path/'checkpoints';server=bridge.make_server(root,TOKEN,port=0,timeout=.5)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield root,f'http://127.0.0.1:{server.server_port}'
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def put(url,path,data=b'model',token=TOKEN,headers=None):
    h={'Authorization':'Bearer '+token,'Content-Length':str(len(data)),bridge.HASH_HEADER:hashlib.sha256(data).hexdigest()}
    h.update(headers or {})
    return urlopen(Request(url+'/files/'+quote(path,safe='/%'),data=data,method='PUT',headers=h))


def test_client_streamed_upload_and_write_only_health(receiver,tmp_path):
    root,url=receiver;source=tmp_path/'head.npy';content=b'checkpoint\0'*200000;source.write_bytes(content)
    result=bridge.upload(url,source,'interception/weights.npy',TOKEN)
    assert result==dict(stored=True,bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
    assert (root/'interception/weights.npy').read_bytes()==content
    assert json.load(urlopen(url+'/health'))=={'ready':True}
    for route in ('/','/files/interception/weights.npy','/list','/token'):
        with pytest.raises(HTTPError) as error:urlopen(url+route)
        assert error.value.code==404


@pytest.mark.parametrize('token',['','wrong','b'*64,'\u00e9'*64])
def test_authentication_before_writes(receiver,token):
    root,url=receiver
    with pytest.raises(HTTPError) as error:put(url,'bad/weights.npy',token=token)
    assert error.value.code==401
    assert list(root.iterdir())==[]


@pytest.mark.parametrize('path',['../outside','x/../outside','/absolute','C:/absolute','C:relative','x\\outside','//server/file','x//file','CON','aux.txt','COM1.npy','LPT9.txt','con .txt','weights.npy:stream','x.','x ','%2e%2e/outside','x%2f..%2foutside','x%5coutside','x%00name'])
def test_path_escape_and_windows_aliases_rejected(receiver,path):
    root,url=receiver
    # Encode transport spaces; the receiver must reject their decoded Win32 aliases.
    with pytest.raises((HTTPError,ValueError)) as error:put(url,path)
    if isinstance(error.value,HTTPError):assert error.value.code==400
    assert not [p for p in root.rglob('*') if p.is_file()]


def test_digest_mismatch_never_changes_target_and_temp_is_removed(receiver):
    root,url=receiver;target=root/'head.npy';target.write_bytes(b'original');old=hashlib.sha256(b'original').hexdigest()
    with pytest.raises(HTTPError) as error:put(url,'head.npy',b'new',headers={bridge.HASH_HEADER:'0'*64,bridge.EXPECTED_HEADER:old})
    assert error.value.code==422 and target.read_bytes()==b'original'
    assert not list(root.glob('.bridge-*'))


def test_overwrite_requires_exact_expected_old_hash(receiver):
    root,url=receiver;put(url,'head.npy',b'old').close()
    for headers in ({},{bridge.EXPECTED_HEADER:'0'*64}):
        with pytest.raises(HTTPError) as error:put(url,'head.npy',b'new',headers=headers)
        assert error.value.code==409 and (root/'head.npy').read_bytes()==b'old'
    with put(url,'head.npy',b'new',headers={bridge.EXPECTED_HEADER:hashlib.sha256(b'old').hexdigest()}) as response:assert response.status==201
    assert (root/'head.npy').read_bytes()==b'new'


def test_symlink_parent_and_target_rejected(receiver,tmp_path):
    root,url=receiver;outside=tmp_path/'outside';outside.mkdir();victim=outside/'victim';victim.write_bytes(b'unchanged')
    try:(root/'escape').symlink_to(outside,target_is_directory=True);(root/'head.npy').symlink_to(victim)
    except OSError:pytest.skip('symlink creation unavailable')
    for path in ('escape/victim','head.npy'):
        with pytest.raises(HTTPError) as error:put(url,path)
        assert error.value.code==400
    assert victim.read_bytes()==b'unchanged'
    with pytest.raises(ValueError):bridge.make_server(root/'escape',TOKEN,port=0)


def test_oversize_and_missing_length_rejected_before_body(receiver):
    root,url=receiver;address=url.split('://')[1];connection=HTTPConnection(address,timeout=2)
    connection.putrequest('PUT','/files/head.npy');connection.putheader('Authorization','Bearer '+TOKEN);connection.putheader(bridge.HASH_HEADER,'0'*64);connection.putheader('Content-Length',str(bridge.MAX_BYTES+1));connection.endheaders()
    assert connection.getresponse().status==413;connection.close();assert not (root/'head.npy').exists()
    connection=HTTPConnection(address,timeout=2);connection.putrequest('PUT','/files/head.npy');connection.putheader('Authorization','Bearer '+TOKEN);connection.putheader(bridge.HASH_HEADER,'0'*64);connection.endheaders()
    assert connection.getresponse().status==400;connection.close()


def test_truncated_body_is_not_committed(receiver):
    root,url=receiver;connection=HTTPConnection(url.split('://')[1],timeout=2)
    connection.putrequest('PUT','/files/head.npy');connection.putheader('Authorization','Bearer '+TOKEN);connection.putheader(bridge.HASH_HEADER,'0'*64);connection.putheader('Content-Length','100');connection.endheaders();connection.send(b'partial')
    assert connection.getresponse().status==408;connection.close()
    assert not (root/'head.npy').exists() and not list(root.glob('.bridge-*'))


def test_token_setup_reuses_secret_without_printing(tmp_path,capsys):
    path=tmp_path/'private'/'token';assert bridge.init_token(path)==path.resolve();first=path.read_text();assert len(first.strip())==64
    assert bridge.init_token(path)==path.resolve() and path.read_text()==first
    assert capsys.readouterr().out==''
    if os.name!='nt':assert path.stat().st_mode&0o777==0o600
    assert bridge.header_token('proxy-secret-placeholder')=='proxy-secret-placeholder'
    with pytest.raises(ValueError):bridge.header_token('secret\r\nInjected: bad')


def test_bounded_reader_and_no_redirect_or_insecure_remote_url(tmp_path):
    reader=bridge.LimitedReader(io.BytesIO(b'123456789'),4);assert reader.read(100)==b'1234' and reader.read()==b''
    assert bridge.NoRedirect().redirect_request(None,None,307,'',{},'https://other.invalid') is None
    source=tmp_path/'head.npy';source.write_bytes(b'head')
    for url in ('http://remote.invalid','https://user:secret@host.invalid','https://host.invalid/?token=x'):
        with pytest.raises(ValueError):bridge.upload(url,source,'head.npy',TOKEN)
