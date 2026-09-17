"""Loopback-only UI/API, independently restartable from the training worker."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse, parse_qs
import uuid

from ..telemetry import Telemetry


def make_server(checkpoint, directory, *, port=8765, evaluation=False, device='cpu', comparison_checkpoints=()):
    checkpoint, directory = Path(checkpoint).resolve(), Path(directory).resolve()
    comparison_checkpoints = [Path(p).resolve() for p in comparison_checkpoints]
    directory.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ('/', '/app.js', '/style.css'):
                name = 'index.html' if path == '/' else path[1:]
                content = (Path(__file__).parent / name).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', {'index.html': 'text/html', 'app.js': 'text/javascript', 'style.css': 'text/css'}[name])
                self.end_headers()
                self.wfile.write(content)
            elif path == '/config':
                self.respond(dict(evaluation=evaluation, checkpoint=checkpoint.name,
                    checkpoints=[p.name for p in [checkpoint, *comparison_checkpoints]]))
            elif path == '/status':
                telemetry = Telemetry(directory)
                try:
                    telemetry.subscribe()
                    self.respond(telemetry.read() or {'waiting': True})
                finally:
                    telemetry.close()
            elif path == '/diagnostics' and evaluation:
                self.respond([p.stem for p in sorted((directory / 'diagnostics').glob('probe-*.json'), key=lambda p: p.stat().st_mtime)])
            elif path == '/diagnostic' and evaluation:
                name = parse_qs(urlparse(self.path).query).get('id', [''])[0]
                p = directory / 'diagnostics' / (name + '.json')
                if not name.startswith('probe-') or any(c not in '0123456789abcdef' for c in name[6:]):
                    self.respond({'error': 'invalid bundle'}, 400)
                else:
                    self.respond(json.loads(p.read_text()) if p.exists() else {})
            elif path in ('/anatomy', '/connection') and evaluation:
                import torch
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                m = payload['metadata']
                graph = m['graph']
                if path == '/anatomy':
                    count = min(len(graph['pre']), 10000)
                    sample = [i * (len(graph['pre']) - 1) // max(1, count - 1) for i in range(count)]
                    self.respond(dict(body_ids=graph['body_ids'], cell_types=m['retina']['cell_types'],
                        positions=m['manifest'].get('positions', []),
                        pre=[graph['pre'][i] for i in sample], post=[graph['post'][i] for i in sample],
                        displayed_edges=min(len(graph['pre']), 10000), total_edges=len(graph['pre'])))
                else:
                    body = int(parse_qs(urlparse(self.path).query).get('body', ['0'])[0])
                    if body not in graph['body_ids']:
                        self.respond({'error': 'unknown body'}, 404)
                        return
                    index = graph['body_ids'].index(body)
                    rows = []
                    for e, (pre, post) in enumerate(zip(graph['pre'], graph['post'])):
                        if pre == index or post == index:
                            rows.append(dict(pre=graph['body_ids'][pre], post=graph['body_ids'][post],
                                contacts=graph['contacts'][e], sign=graph['signs'][pre],
                                magnitude=float(payload['state']['network']['magnitudes'][e])))
                    self.respond({'connections': rows[:200], 'total': len(rows)})
            else:
                self.respond({'error': 'not found'}, 404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if origin is not None and origin not in allowed:
                self.respond({'error': 'foreign origin'}, 403)
                return
            if self.headers.get('Content-Type') != 'application/json':
                self.respond({'error': 'JSON required'}, 415)
                return
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 8192:
                self.respond({'error': 'invalid command size'}, 400)
                return
            try:
                command = json.loads(self.rfile.read(length))
                if self.path == '/start':
                    args = [sys.executable, '-m', 'fly_connectome.worker', str(checkpoint),
                            '--directory', str(directory), '--device', device]
                    if evaluation:
                        args.append('--evaluation')
                        for source in comparison_checkpoints:
                            args.extend(['--comparison-checkpoint', str(source)])
                    with (directory / 'worker.log').open('wb') as output:
                        subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                            start_new_session=os.name != 'nt')
                    self.respond({'started': True})
                elif self.path == '/command':
                    actions = {'pause', 'resume', 'stop', 'reset', 'save', 'restart'}
                    if evaluation:
                        actions = {'pause', 'resume', 'stop', 'reset', 'step', 'control', 'human', 'speed', 'probe'}
                    if command.get('action') not in actions:
                        raise ValueError('unsupported action')
                    if len(list(directory.glob('command-*.json'))) >= 32:
                        self.respond({'error': 'command queue full'}, 429)
                        return
                    path = directory / f'command-{time.time_ns():020}-{uuid.uuid4().hex}.json'
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(command))
                    temporary.replace(path)
                    self.respond({'queued': True})
                else:
                    self.respond({'error': 'not found'}, 404)
            except (ValueError, KeyError, TypeError) as error:
                self.respond({'error': str(error)}, 400)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--directory', default='runs/ui')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--comparison-checkpoint', action='append', default=[])
    args = parser.parse_args()
    server = make_server(args.checkpoint, args.directory, port=args.port, evaluation=args.evaluation, device=args.device,
                         comparison_checkpoints=args.comparison_checkpoint)
    print(f'Local UI: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
