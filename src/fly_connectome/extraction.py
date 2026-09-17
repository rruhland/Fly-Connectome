"""Preregistered MaleCNS subgraph extraction using annotated neurons and real ROIs."""
from dataclasses import dataclass
import re
import numpy as np

from .anatomy import select_roster, choose_threshold, anatomy_report
from .graph import Graph


@dataclass(frozen=True)
class Selection:
    optic_side: str = 'R'
    max_hops: int = 8
    path_threshold: int = 3
    reciprocal_contacts: int = 20
    scc_cap: int = 128
    minimum_nt_confidence: float = .5
    gain: float = .005
    required_types: tuple = ('L1', 'L2', 'L3', 'L4', 'T4', 'T5', 'LPi', 'LC10', 'DNa02')
    required_regions: tuple = ('LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AOTU(R)', 'PVLP(R)', 'PLP(R)')


@dataclass
class Extracted:
    graph: Graph
    threshold: int
    pathways: list
    motor_up: list
    motor_down: list
    retina: dict
    regions: list
    positions: list
    sign_evidence: list
    reports: dict
    stages: list


def _sign(row, confidence):
    mapping = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1, 'histamine': -1}
    nt = row.get('ground_truth')
    if nt in mapping:
        return mapping[nt], dict(source='ground_truth', neurotransmitter=nt)
    for nt_field, confidence_field in (('predicted_nt', 'predicted_nt_confidence'),
                                      ('celltype_predicted_nt', 'celltype_predicted_nt_confidence')):
        nt, score = row.get(nt_field), row.get(confidence_field)
        if nt in mapping and score is not None and score >= confidence:
            return mapping[nt], dict(source=nt_field, neurotransmitter=nt, confidence=score)
    return None


def _stage(cell_type, superclass):
    cell_type = cell_type or 'untyped'
    if cell_type == 'DNa02':
        return 5
    if re.fullmatch(r'L[1-4]', cell_type):
        return 0
    if cell_type.startswith(('T4', 'T5')):
        return 2
    if cell_type.startswith(('LPi', 'LC10')) or superclass == 'visual_projection':
        return 3
    if superclass == 'ol_intrinsic':
        return 1
    return 4


def extract_tables(annotations, neurotransmitters, roi_by_id, edges, rules=Selection()):
    nt_by_id = {r['body']: r for r in neurotransmitters}
    evidence, candidates = {}, {}
    for row in annotations:
        body, cell_type = row['bodyId'], row.get('type') or ''
        if row.get('status') != 'Traced' or cell_type in ('L5', 'LC11'):
            continue
        superclass = row.get('superclass')
        rois = roi_by_id.get(body, {})
        central_roi = any(f'{r}({s})' in rois for r in ('AOTU', 'PVLP', 'PLP') for s in ('R', 'L'))
        visual = superclass in ('ol_intrinsic', 'visual_projection', 'visual_centrifugal') and row.get('somaSide') == rules.optic_side
        central = superclass == 'cb_intrinsic' and central_roi
        if not (visual or central or cell_type == 'DNa02'):
            continue
        sign = _sign(nt_by_id.get(body, {}), rules.minimum_nt_confidence)
        if sign is not None:
            candidates[body] = row
            evidence[body] = sign
    body_ids = sorted(candidates)
    retained = np.isin(edges[:, 0], body_ids) & np.isin(edges[:, 1], body_ids)
    edges = edges[retained]
    candidate = Graph.from_contacts(body_ids, edges[:, 0], edges[:, 1], edges[:, 2],
                                     [evidence[b][0] for b in body_ids], rules.gain)
    seed_pattern = r'(L[1-4]|T[45][a-d]?|LPi.*|LC10[a-z]?|HS.*|VS.*|H1|H2|CH|VCH|DCH)'
    seeds = [b for b in body_ids if re.fullmatch(seed_pattern, candidates[b]['type'] or '')]
    motors = [b for b in body_ids if candidates[b]['type'] == 'DNa02']
    selected = select_roster(candidate.threshold(rules.path_threshold), seeds, motors,
        max_hops=rules.max_hops, reciprocal_contacts=rules.reciprocal_contacts, scc_cap=rules.scc_cap)
    cell_types = [candidates[b]['type'] or 'untyped' for b in selected]
    for required in rules.required_types:
        if not any(t == required or (required in ('T4', 'T5', 'LPi', 'LC10') and t.startswith(required)) for t in cell_types):
            raise ValueError(f"missing required population: {required}")
    retained = np.isin(edges[:, 0], selected) & np.isin(edges[:, 1], selected)
    edges = edges[retained]
    graph = Graph.from_contacts(selected, edges[:, 0], edges[:, 1], edges[:, 2],
                               [evidence[b][0] for b in selected], rules.gain)
    sensory = [b for b in selected if candidates[b]['type'] in ('L1', 'L2', 'L3')]
    regions = [sorted(roi_by_id.get(b, {})) for b in selected]
    threshold = choose_threshold(graph, sensory, motors, regions, rules.required_regions)
    stages = [_stage(candidates[b]['type'], candidates[b].get('superclass')) for b in selected]
    pathways = ['behavioral' if stages[j] >= 4 or stages[i] >= 4 else
                ('feedforward' if stages[i] < stages[j] or
                 (candidates[selected[i]]['type'] == 'L2' and candidates[selected[j]]['type'] == 'L4') else 'predictive')
                for i, j in zip(graph.pre, graph.post)]
    columns = {}
    neuron_columns = []
    for b in selected:
        row = candidates[b]
        if b not in sensory:
            neuron_columns.append(-1)
            continue
        xy = (row.get('assignedOlHex1'), row.get('assignedOlHex2'))
        if any(v is None or not np.isfinite(v) for v in xy):
            raise ValueError(f"sensory body {b} lacks assigned retinotopic column")
        if xy not in columns:
            columns[xy] = len(columns)
        neuron_columns.append(columns[xy])
    up = [b for b in motors if candidates[b].get('somaSide') == 'R']
    down = [b for b in motors if candidates[b].get('somaSide') == 'L']
    if not up or not down:
        raise ValueError("both opponent DNa02 populations required")
    retina = dict(height=32, width=64, hex_columns=[list(xy) for xy in columns],
                  neuron_columns=neuron_columns, cell_types=cell_types,
                  injection={'L1': 'on', 'L2': 'off', 'L3': 'off'})
    reports = {str(t): anatomy_report(graph.threshold(t), sensory, motors, regions) for t in (1, 3, 5)}
    return Extracted(graph, threshold, pathways, up, down, retina, regions,
                     [candidates[b].get('somaLocation') for b in selected],
                     [dict(body_id=b, **evidence[b][1]) for b in selected], reports, stages)
