"""Authenticated write-only artifact receiver and HTTPS upload client (stdlib only).

Receiver: loopback only; expose privately with Tailscale Serve, never Funnel.
Client: PONG_BRIDGE_TOKEN environment secret or --token-file, never a token argument.
"""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_BYTES = 64 * 1024 * 1024
TOKEN_ENV = 'PONG_BRIDGE_TOKEN'
HASH_HEADER = 'X-Content-SHA256'
EXPECTED_HEADER = 'X-Expected-Existing-SHA256'
_HEX = re.compile(r'[0-9a-fA-F]{64}\Z')
_PORTABLE = re.compile(r'[A-Za-z0-9_. -]+\Z')
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}


def default_token_file():
    base = Path(os.environ['LOCALAPPDATA']) if os.environ.get('LOCALAPPDATA') else Path.home() / '.config'
    return base / 'FlyConnectomeBridge' / 'token'


def validate_token(value):
    value = value.strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{43,128}', value):
        raise ValueError('token file/environment must contain a valid bridge token')
    return value


def header_token(value):
    # Cloud runtime proxies may substitute the real secret for this opaque value.
    # Keep it untouched except surrounding whitespace; never bypass the proxy.
    value = value.strip()
    if not value or any(ord(c) < 32 or ord(c) > 126 for c in value):
        raise ValueError('nonempty header-safe token configuration is required')
    return value


def read_token(path):
    path = Path(path)
    if path.stat().st_size > 256:
        raise ValueError('invalid token file')
    return validate_token(path.read_text(encoding='ascii'))


def init_token(path):
    """Create 256 random bits outside the repository; reuse a valid existing file."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        read_token(path)
    else:
        with os.fdopen(fd, 'w', encoding='ascii') as stream:
            stream.write(secrets.token_hex(32) + '\n')
    if os.name != 'nt':
        path.chmod(0o600)
    return path.resolve()


def relative_parts(value):
    """Portable filenames only: reject traversal, ADS, drives and Win32 aliases."""
    if not isinstance(value, str) or not 1 <= len(value) <= 240:
        raise ValueError('invalid relative artifact path')
    parts = value.split('/')
    if len(parts) > 12:
        raise ValueError('artifact path is too deep')
    for part in parts:
        if (part in ('', '.', '..') or len(part) > 120 or not _PORTABLE.fullmatch(part)
                or part.endswith(('.', ' ')) or part.startswith(' ')
                or part.split('.')[0].rstrip(' ').upper() in _RESERVED):
            raise ValueError('invalid portable artifact path')
    return parts


def _check_plain(path, directory=False):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('symlinks and reparse points are not permitted')
    if directory and not stat.S_ISDIR(info.st_mode):
        raise ValueError('artifact parent is not a directory')
    if not directory and not stat.S_ISREG(info.st_mode):
        raise ValueError('artifact target is not a regular file')
    return True


def checked_target(root, relative, create=False):
    parts = relative_parts(relative)
    _check_plain(root, directory=True)
    current = root
    for part in parts[:-1]:
        current = current / part
        exists = _check_plain(current, directory=True)
        if not exists and create:
            current.mkdir(mode=0o700, exist_ok=True)
            _check_plain(current, directory=True)
    target = current / parts[-1]
    _check_plain(target)
    if not target.resolve().is_relative_to(root):
        raise ValueError('artifact path escapes the storage root')
    return target


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(128 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Conflict(Exception):
    pass


class DigestMismatch(Exception):
    pass


def make_server(root, token, port=8769, timeout=20):
    root = Path(root).expanduser().absolute()
    if root.exists():
        _check_plain(root, directory=True)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    token = validate_token(token)
    commit_lock = threading.Lock()

    def check_existing(target, expected):
        exists = _check_plain(target)
        if exists:
            if expected is None:
                raise Conflict('target exists; explicit expected existing SHA256 required')
            if target.stat().st_size > MAX_BYTES or not hmac.compare_digest(file_sha256(target), expected):
                raise Conflict('existing target does not match the expected SHA256')
        elif expected is not None:
            raise Conflict('expected existing target is missing')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log authorization, request bodies, or URL input.

        def respond(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == '/health':
                self.respond({'ready': True})
            else:
                self.respond({'error': 'not found'}, 404)

        def do_PUT(self):
            authorization = self.headers.get_all('Authorization', [])
            try:
                valid = len(authorization) == 1 and hmac.compare_digest(authorization[0].encode('ascii'), ('Bearer ' + token).encode('ascii'))
            except UnicodeEncodeError:
                valid = False
            if not valid:
                return self.respond({'error': 'unauthorized'}, 401)
            temporary = None
            target = None
            try:
                self.connection.settimeout(timeout)
                url = urlsplit(self.path)
                if url.query or url.fragment or not url.path.startswith('/files/'):
                    raise ValueError('PUT requires /files/<relative-path>')
                relative = unquote(url.path[len('/files/'):], errors='strict')
                lengths = self.headers.get_all('Content-Length', [])
                hashes = self.headers.get_all(HASH_HEADER, [])
                expected_values = self.headers.get_all(EXPECTED_HEADER, [])
                if self.headers.get('Transfer-Encoding') or len(lengths) != 1 or not re.fullmatch(r'\d{1,12}', lengths[0]):
                    raise ValueError('one finite Content-Length is required')
                length = int(lengths[0])
                if length > MAX_BYTES:
                    return self.respond({'error': 'upload exceeds 64 MiB'}, 413)
                if len(hashes) != 1 or not _HEX.fullmatch(hashes[0]):
                    raise ValueError('valid content SHA256 header required')
                if len(expected_values) > 1 or expected_values and not _HEX.fullmatch(expected_values[0]):
                    raise ValueError('invalid expected existing SHA256')
                expected = expected_values[0].lower() if expected_values else None
                with commit_lock:
                    target = checked_target(root, relative, create=True)
                    check_existing(target, expected)
                fd, temporary_name = tempfile.mkstemp(prefix='.bridge-', dir=target.parent)
                temporary = Path(temporary_name)
                digest = hashlib.sha256()
                remaining = length
                body_deadline = time.monotonic() + timeout
                with os.fdopen(fd, 'wb') as stream:
                    while remaining:
                        left = body_deadline - time.monotonic()
                        if left <= 0:
                            raise TimeoutError('body deadline')
                        self.connection.settimeout(left)
                        block = self.rfile.read1(min(128 * 1024, remaining))
                        if not block:
                            raise ValueError('incomplete upload body')
                        stream.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                actual = digest.hexdigest()
                if not hmac.compare_digest(actual, hashes[0].lower()):
                    raise DigestMismatch()
                with commit_lock:
                    # Recheck parent links and compare-and-swap immediately before
                    # atomic replacement. Concurrent uploads cannot bypass the guard.
                    current = checked_target(root, relative)
                    if current != target:
                        raise ValueError('artifact target changed during upload')
                    check_existing(target, expected)
                    os.replace(temporary, target)
                    temporary = None
                response, status = {'stored': True, 'bytes': length, 'sha256': actual}, 201
            except Conflict as error:
                response, status = {'error': str(error)}, 409
            except DigestMismatch:
                response, status = {'error': 'content SHA256 mismatch; target unchanged'}, 422
            except (ValueError, UnicodeError):
                response, status = {'error': 'invalid path, headers, or incomplete body'}, 400
            except TimeoutError:
                response, status = {'error': 'upload timed out; target unchanged'}, 408
            except OSError:
                response, status = {'error': 'storage operation failed; target not committed'}, 500
            finally:
                if temporary is not None:
                    try:
                        # Do not follow a parent replaced by a local symlink.
                        if target is not None and checked_target(root, relative).parent == temporary.parent:
                            temporary.unlink(missing_ok=True)
                    except (OSError, ValueError):
                        pass
            self.respond(response, status)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # A bearer credential must never follow a redirected upload.


class LimitedReader:
    def __init__(self, stream, remaining):
        self.stream = stream
        self.remaining = remaining

    def read(self, size=-1):
        size = min(self.remaining, 128 * 1024 if size < 0 else size)
        if not size:
            return b''
        data = self.stream.read(size)
        self.remaining -= len(data)
        return data


def upload(url, source, relative, token, expected=None, timeout=60):
    relative_parts(relative)
    parsed = urlsplit(url)
    if (parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/')
            or parsed.scheme not in ('https', 'http') or not parsed.hostname):
        raise ValueError('use the receiver base HTTPS URL without credentials, path or query')
    if parsed.scheme == 'http' and parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('HTTPS is required except for local loopback testing')
    token = header_token(token)
    if expected is not None and not _HEX.fullmatch(expected):
        raise ValueError('expected existing SHA256 must be 64 hexadecimal characters')
    source = Path(source)
    size = source.stat().st_size
    if size > MAX_BYTES or not source.is_file():
        raise ValueError('upload source must be a regular file no larger than 64 MiB')
    digest = file_sha256(source)
    headers = {'Authorization': 'Bearer ' + token, 'Content-Length': str(size),
               'Content-Type': 'application/octet-stream', HASH_HEADER: digest}
    if expected is not None:
        headers[EXPECTED_HEADER] = expected.lower()
    # urllib's default proxy and verified TLS handlers remain enabled.
    opener = build_opener(NoRedirect())
    with source.open('rb') as stream:
        request = Request(url.rstrip('/') + '/files/' + quote(relative, safe='/'), data=LimitedReader(stream, size), headers=headers, method='PUT')
        with opener.open(request, timeout=timeout) as response:
            receipt = json.loads(response.read(8192))
    if receipt != {'stored': True, 'bytes': size, 'sha256': digest}:
        raise ValueError('receiver acknowledgement did not match the upload')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init-token', help='create/reuse local token; print location only')
    init.add_argument('--token-file', type=Path, default=default_token_file())
    receive = sub.add_parser('receive', help='run write-only receiver on 127.0.0.1')
    receive.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'checkpoints')
    receive.add_argument('--token-file', type=Path, default=default_token_file())
    receive.add_argument('--port', type=int, default=8769)
    send = sub.add_parser('send', help='stream one file; token comes from environment or file')
    send.add_argument('--url', default=os.environ.get('PONG_BRIDGE_URL'))
    send.add_argument('--file', type=Path, required=True)
    send.add_argument('--path', required=True, help='relative destination inside receiver root')
    send.add_argument('--token-file', type=Path)
    send.add_argument('--expected-existing-sha256')
    args = parser.parse_args()
    try:
        if args.command == 'init-token':
            print('Token file ready (value is not displayed): ' + str(init_token(args.token_file)))
        elif args.command == 'receive':
            server = make_server(args.root, read_token(args.token_file), args.port)
            print(f'Artifact receiver ready at http://127.0.0.1:{server.server_port}; private Tailscale Serve only.', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        else:
            if not args.url:
                raise ValueError('--url or PONG_BRIDGE_URL is required')
            token = read_token(args.token_file) if args.token_file else header_token(os.environ.get(TOKEN_ENV, ''))
            result = upload(args.url, args.file, args.path, token, args.expected_existing_sha256)
            print(json.dumps(dict(path=args.path, **result)))
    except HTTPError as error:
        parser.exit(1, f'Upload rejected: HTTP {error.code}. No credentials or response body displayed.\n')
    except (OSError, ValueError, URLError):
        parser.exit(1, 'Bridge failed: check the local paths, token configuration and private network connection. No secret value displayed.\n')


if __name__ == '__main__':
    main()
