"""Small structurally evolved programs over anonymous observed object pairs.

This module contains no simulator access, training labels, or reward estimator.
Only the external complete-game scalar-reward search selects programs.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'anonymous-arithmetic-residual-program-v1'
MAX_NODES = 16
MAX_OBJECTS = 8
CONSTANTS = (-2., -1., -.5, 0., .5, 1., 2.)
GAINS = (.25, .5, 1., 2., 4.)
OPS = ('add', 'sub', 'mul', 'div', 'abs', 'min', 'max', 'lt', 'select')
ARITIES = (2, 2, 2, 2, 1, 2, 2, 2, 3)
POOLS = ('mean', 'min', 'max')
N_TERMINALS = 12 + 5 + len(CONSTANTS)
ZERO = 17 + CONSTANTS.index(0.)
CONFIG = dict(schema=SCHEMA, max_nodes=MAX_NODES, max_objects=MAX_OBJECTS,
    terminals='ordered_including_self_pairs_2x[y,x,vy,vx,observed,known]+phase_onehot+constants',
    position_scale=2., velocity_scale=1., constants=CONSTANTS, gains=GAINS,
    operators=OPS, arities=ARITIES, pools=POOLS, node_clip=8., output_clip=1.,
    division_floor=1/16, score_range_floor=1e-8,
    before_first_actual_hit='exact_baseline', unsupported_scene='exact_baseline',
    recurrence=False, teacher_action_labels=False)
CONFIG_JSON = json.dumps(CONFIG, sort_keys=True, separators=(',', ':'))


def _different(rng, old, count):
    """Uniformly choose a different categorical value."""
    value = int(rng.integers(count - 1))
    return value + int(value >= old)


def _scene(scene):
    rows = np.asarray(scene, dtype=np.float64)
    if rows.size == 0:
        return np.empty((0, 6), np.float64)
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError('finite anonymous six-field rows required')
    if not np.isin(rows[:, 4:], (0., 1.)).all():
        raise ValueError('observed and velocity-known flags must be binary')
    return rows[np.lexsort(rows[:, ::-1].T)]


@dataclass(frozen=True)
class Program:
    """Immutable feedforward DAG; references below N_TERMINALS name inputs."""
    nodes: tuple
    outputs: tuple
    pools: tuple
    gain: float

    def __post_init__(self):
        nodes = tuple(tuple(int(v) for v in node) for node in self.nodes)
        outputs = tuple(int(v) for v in self.outputs)
        pools = tuple(int(v) for v in self.pools)
        gain = float(self.gain)
        if len(nodes) != MAX_NODES or any(len(n) != 4 for n in nodes):
            raise ValueError('exactly16 four-gene nodes required')
        for i, (op, *args) in enumerate(nodes):
            if not 0 <= op < len(OPS) or any(not 0 <= a < N_TERMINALS + i for a in args):
                raise ValueError('invalid operator or non-feedforward edge')
        if len(outputs) != 3 or any(not 0 <= v < N_TERMINALS + MAX_NODES for v in outputs):
            raise ValueError('exactly three valid output references required')
        if len(pools) != 3 or any(not 0 <= v < len(POOLS) for v in pools):
            raise ValueError('exactly three valid pooling choices required')
        if gain not in (0., *GAINS):
            raise ValueError('gain outside fixed alphabet')
        object.__setattr__(self, 'nodes', nodes)
        object.__setattr__(self, 'outputs', outputs)
        object.__setattr__(self, 'pools', pools)
        object.__setattr__(self, 'gain', gain)

    @classmethod
    def random(cls, rng):
        nodes = tuple((int(rng.integers(len(OPS))),
            *(int(rng.integers(N_TERMINALS + i)) for _ in range(3)))
            for i in range(MAX_NODES))
        return cls(nodes, tuple(int(rng.integers(N_TERMINALS + MAX_NODES)) for _ in range(3)),
            tuple(int(rng.integers(len(POOLS))) for _ in range(3)), float(rng.choice(GAINS)))

    @classmethod
    def zero(cls):
        # Deterministic dormant genetic material; no output or active inference.
        base = cls.random(np.random.default_rng(619217))
        return cls(base.nodes, (ZERO,) * 3, (0,) * 3, 0.)

    def active_indices(self):
        if self.gain == 0:
            return ()
        active = set()
        def visit(ref):
            if ref < N_TERMINALS or ref - N_TERMINALS in active:
                return
            i = ref - N_TERMINALS
            active.add(i)
            op, *args = self.nodes[i]
            for arg in args[:ARITIES[op]]:
                visit(arg)
        for ref in self.outputs:
            visit(ref)
        return tuple(sorted(active))

    @property
    def active_node_count(self):
        return len(self.active_indices())

    @property
    def complexity(self):
        return self.active_node_count

    def _active_expression(self):
        def expression(ref):
            if ref < N_TERMINALS:
                return ('terminal', ref)
            op, *args = self.nodes[ref - N_TERMINALS]
            return (OPS[op], *(expression(a) for a in args[:ARITIES[op]]))
        return (self.gain, tuple((POOLS[p], expression(o)) for p, o in zip(self.pools, self.outputs)))

    def active_hash(self):
        expression = (0.,) if self.gain == 0 else self._active_expression()
        payload = json.dumps((CONFIG_JSON, expression), separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest()

    def frozen_parameters(self):
        return (SCHEMA, CONFIG_JSON, self.nodes, self.outputs, self.pools, self.gain)

    def to_dict(self):
        return dict(schema=SCHEMA, config=json.loads(CONFIG_JSON), nodes=[list(n) for n in self.nodes],
            outputs=list(self.outputs), pools=list(self.pools), gain=self.gain,
            active_hash=self.active_hash(), active_node_count=self.active_node_count)

    @classmethod
    def from_dict(cls, payload):
        if payload.get('schema') != SCHEMA or json.dumps(payload.get('config'), sort_keys=True,
                separators=(',', ':')) != CONFIG_JSON:
            raise ValueError('program grammar/config mismatch')
        program = cls(payload['nodes'], payload['outputs'], payload['pools'], payload['gain'])
        if payload.get('active_hash', program.active_hash()) != program.active_hash():
            raise ValueError('active program hash mismatch')
        return program

    def mutate(self, rng):
        """Change1–3 active structural genes; optionally change the discrete gain.

        Inactive-only edits and gain-only changes are never accepted as challengers.
        """
        original_hash = self.active_hash()
        original_structure = self._active_expression()[1]
        for _ in range(64):
            nodes = [list(n) for n in self.nodes]
            outputs, pools = list(self.outputs), list(self.pools)
            gain = float(rng.choice(GAINS)) if self.gain == 0 or rng.random() < .25 else self.gain
            # Zero incumbent must activate an output; dormant code is not fitted.
            if self.gain == 0:
                head = int(rng.integers(3))
                outputs[head] = _different(rng, outputs[head], N_TERMINALS + MAX_NODES)
            current = Program(nodes, outputs, pools, gain)
            for _ in range(int(rng.integers(1, 4))):
                fields = [('output', h, 0) for h in range(3)] + [('pool', h, 0) for h in range(3)]
                for i in current.active_indices():
                    fields.append(('node', i, 0))
                    fields.extend(('node', i, a + 1) for a in range(ARITIES[nodes[i][0]]))
                kind, i, j = fields[int(rng.integers(len(fields)))]
                if kind == 'output':
                    outputs[i] = _different(rng, outputs[i], N_TERMINALS + MAX_NODES)
                elif kind == 'pool':
                    pools[i] = _different(rng, pools[i], len(POOLS))
                else:
                    count = len(OPS) if j == 0 else N_TERMINALS + i
                    nodes[i][j] = _different(rng, nodes[i][j], count)
                current = Program(nodes, outputs, pools, gain)
            if (current.active_hash() != original_hash
                    and current._active_expression()[1] != original_structure):
                return current
        raise RuntimeError('could not produce an active structural mutation')

    def residual(self, scene, phase):
        if not isinstance(phase, (int, np.integer)) or not 0 <= int(phase) < 5:
            raise ValueError('public camera phase must be an integer in[0,5)')
        rows = _scene(scene)
        if self.gain == 0 or not 0 < len(rows) <= MAX_OBJECTS:
            return np.zeros(3, np.float64)
        rows = rows.copy()
        rows[:, :2] *= 2.
        rows[:, :4] = np.clip(rows[:, :4], -8., 8.)
        i, j = np.indices((len(rows), len(rows))).reshape(2, -1)
        pair = np.column_stack((rows[i], rows[j]))
        context = np.r_[np.eye(5)[int(phase)], CONSTANTS]
        values = [pair[:, k] for k in range(12)]
        values.extend(np.full(len(pair), v) for v in context)
        values.extend([None] * MAX_NODES)
        for k in self.active_indices():
            op, *args = self.nodes[k]
            a = values[args[0]]
            b = values[args[1]] if ARITIES[op] >= 2 else None
            if op == 0: value = a + b
            elif op == 1: value = a - b
            elif op == 2: value = a * b
            elif op == 3:
                denominator = np.where(np.abs(b) < 1/16, np.where(b < 0, -1/16, 1/16), b)
                value = a / denominator
            elif op == 4: value = np.abs(a)
            elif op == 5: value = np.minimum(a, b)
            elif op == 6: value = np.maximum(a, b)
            elif op == 7: value = (a < b).astype(np.float64)
            else: value = np.where(a > 0, b, values[args[2]])
            values[N_TERMINALS + k] = np.clip(value, -8., 8.)
        reduced = []
        for ref, pool in zip(self.outputs, self.pools):
            value = values[ref]
            reduced.append(np.mean(value) if pool == 0 else np.min(value) if pool == 1 else np.max(value))
        return np.clip(np.asarray(reduced, np.float64), -1., 1.)

    def choose(self, scene, baseline_q, phase, past_hit):
        q = np.asarray(baseline_q, np.float64)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError('three finite baseline scores required')
        q = q.copy()
        if past_hit and self.gain:
            q += self.gain * max(float(np.ptp(q)), 1e-8) * self.residual(scene, phase)
        return int(np.argmax(q)), q

    def action_signature(self, observations):
        """Deduplicate actions only; never attach saved returns to this signature."""
        digest = hashlib.sha256()
        for scene, q, phase, past_hit in observations:
            action, _ = self.choose(scene, q, phase, past_hit)
            digest.update(bytes((action,)))
        return digest.hexdigest()


def save_program(path, program):
    Path(path).write_text(json.dumps(program.to_dict(), sort_keys=True, indent=2) + '\n')


def load_program(path):
    return Program.from_dict(json.loads(Path(path).read_text()))
