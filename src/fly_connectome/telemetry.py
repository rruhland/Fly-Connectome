"""Bounded latest-value shared-memory telemetry; producer never waits for readers."""
import json
import mmap
from pathlib import Path
import struct
import time

SIZE = 65536
HEADER = 24


class Telemetry:
    def __init__(self, directory):
        self.path = Path(directory) / 'telemetry.bin'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open('wb') as stream:
                stream.truncate(SIZE)
        self.file = self.path.open('r+b')
        self.memory = mmap.mmap(self.file.fileno(), SIZE)

    def subscribe(self, now=None):
        now = time.monotonic() if now is None else now
        struct.pack_into('<d', self.memory, 0, now + 3.)

    def enabled(self, now=None):
        now = time.monotonic() if now is None else now
        return struct.unpack_from('<d', self.memory, 0)[0] > now

    def publish(self, capture):
        if not self.enabled():
            return False
        payload = json.dumps(capture(), allow_nan=False, separators=(',', ':')).encode()
        if len(payload) > SIZE - HEADER:
            return False
        generation = struct.unpack_from('<Q', self.memory, 8)[0]
        struct.pack_into('<Q', self.memory, 8, generation + 1)
        self.memory[HEADER:HEADER + len(payload)] = payload
        struct.pack_into('<I', self.memory, 16, len(payload))
        struct.pack_into('<Q', self.memory, 8, generation + 2)
        return True

    def read(self):
        generation = struct.unpack_from('<Q', self.memory, 8)[0]
        if generation % 2:
            return None
        length = struct.unpack_from('<I', self.memory, 16)[0]
        if not 0 < length <= SIZE - HEADER:
            return None
        data = self.memory[HEADER:HEADER + length]
        if struct.unpack_from('<Q', self.memory, 8)[0] != generation:
            return None
        try:
            return json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return None

    def close(self):
        self.memory.close()
        self.file.close()


def latest_snapshot(directory):
    telemetry = Telemetry(directory)
    try:
        return telemetry.read()
    finally:
        telemetry.close()
