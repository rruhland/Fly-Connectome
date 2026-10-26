"""Run this Git checkout's viewer with separately stored inference checkpoints."""
import argparse
from pathlib import Path
import sys
import threading
import webbrowser


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoints', type=Path, default=repo / 'checkpoints',
        help='Checkpoint directory from the downloaded viewer (default: repo/checkpoints).',
    )
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    checkpoints = args.checkpoints.expanduser().resolve()
    if not (checkpoints / 'pong-interception-checkpoint-20261025' / 'checkpoint.json').is_file():
        parser.error('Checkpoint bundle not found. Pass --checkpoints PATH pointing to '
                     'the checkpoints folder in your extracted pong-viewer-local download.')
    # Always use this checkout's code, even if another version is installed.
    sys.path.insert(0, str(repo / 'src'))
    from fly_connectome.viewer.server import make_server

    server = make_server(checkpoints, host='127.0.0.1', port=args.port)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Frozen Pong viewer: {url}\nCode: {repo}\nCheckpoints: {checkpoints}\n'
          'Keep this terminal open; Ctrl+C stops the viewer.', flush=True)
    if not args.no_browser:
        timer = threading.Timer(.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
