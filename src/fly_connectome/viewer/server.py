"""Serve the isolated frozen-checkpoint Pong viewer; never starts training."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import traceback
from pathlib import Path
from urllib.parse import urlparse
import torch
from .catalog import load_catalog
from .engine import ViewerEngine


def make_server(experiments,export=None,*,host='127.0.0.1',port=8767):
    torch.set_num_threads(1)
    catalog,unavailable=load_catalog(experiments,export);engine=ViewerEngine(catalog)
    assets=Path(__file__).parent
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,body,status=200,content_type='application/json'):
            if not isinstance(body,bytes):body=json.dumps(body,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            path=urlparse(self.path).path
            if path in ('/','/app.js','/style.css'):
                name='index.html' if path=='/' else path[1:]
                self.send((assets/name).read_bytes(),content_type={'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name])
            elif path=='/api/catalog':
                self.send(dict(checkpoints=[c.public() for c in catalog.values()],default='final',unavailable=unavailable,
                    description='Frozen learned policies. Viewer episodes are unscored demonstrations, separate from the recorded benchmarks. No training or checkpoint writes.'))
            elif path=='/health':self.send(dict(ok=True,mode='frozen-viewer',checkpoints=len(catalog)))
            else:self.send(dict(error='not found'),404)
        def do_POST(self):
            try:
                origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc!=self.headers.get('Host'):
                    return self.send(dict(error='same-origin viewer requests required'),403)
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send(dict(error='JSON required'),415)
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8192:return self.send(dict(error='invalid request size'),400)
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise ValueError('JSON object required')
                if self.path=='/api/session':
                    identity,state=engine.create();self.send(dict(session=identity,state=state))
                elif self.path=='/api/command':
                    self.send(dict(state=engine.command(data.get('session'),data)))
                else:self.send(dict(error='not found'),404)
            except (ValueError,TypeError,KeyError) as error:self.send(dict(error=str(error)),400)
            except Exception:
                traceback.print_exc();self.send(dict(error='Viewer could not complete this request. Reload to start a clean session.'),500)
    server=ThreadingHTTPServer((host,port),Handler);server.engine=engine;return server


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiments',default='/workspace/m1b-experiments');p.add_argument('--export');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8767);a=p.parse_args()
    server=make_server(a.experiments,a.export,host=a.host,port=a.port)
    print(f'Frozen Pong viewer: http://{a.host}:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        for session in server.engine.sessions.values():session.evaluator.verify_frozen()
        server.server_close()


if __name__=='__main__':main()
