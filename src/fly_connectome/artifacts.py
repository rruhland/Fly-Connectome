"""Construct M1A/M1B runs from an extracted, hashed measured graph."""
import json
from pathlib import Path
import numpy as np

from .graph import Graph
from .sensor import Retina
from .training import Trainer, RunConfig, load_checkpoint, warm_start


def initialize(directory, *, stage='M1A', seeds=(1,), threshold=None, device='cpu', warm_checkpoint=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    with np.load(directory / 'graph-t1.npz', allow_pickle=False) as arrays:
        full = Graph(**{name: arrays[name] for name in ('body_ids', 'pre', 'post', 'contacts', 'signs')}, gain=float(arrays['gain']))
    if full.identity() != manifest['graph_sha256']:
        raise ValueError("extracted graph checksum mismatch")
    threshold = manifest['threshold'] if threshold is None else threshold
    if threshold not in (1, 3, 5):
        raise ValueError("threshold must be T1, T3 or T5")
    node_mask = np.asarray(manifest['stages']) < 4 if stage == 'M1A' else np.ones(len(full.body_ids), dtype=bool)
    edge_mask = node_mask[full.pre] & node_mask[full.post] & (full.contacts >= threshold)
    ids = full.body_ids[node_mask]
    graph = Graph.from_contacts(ids, full.body_ids[full.pre[edge_mask]], full.body_ids[full.post[edge_mask]],
                                full.contacts[edge_mask], full.signs[node_mask], full.gain)
    retina_spec = dict(manifest['retina'])
    for field in ('cell_types', 'neuron_columns'):
        retina_spec[field] = [v for v, keep in zip(retina_spec[field], node_mask) if keep]
    pathways = [p for p, keep in zip(manifest['pathways'], edge_mask) if keep]
    manifest = dict(manifest, source_graph_sha256=full.identity(), graph_sha256=graph.identity(),
                    threshold=threshold, stage=stage, body_ids=ids.tolist(), pathways=pathways, retina=retina_spec)
    for field in ('positions', 'regions', 'stages', 'sign_evidence'):
        if field in manifest:
            manifest[field] = [v for v, keep in zip(manifest[field], node_mask) if keep]
    up = manifest['motor_up'] if stage == 'M1B' else []
    down = manifest['motor_down'] if stage == 'M1B' else []
    model = Trainer(graph, [1] * len(graph.pre), pathways, Retina(**retina_spec, device=device),
                    up, down, list(seeds), config=RunConfig(stage=stage), manifest=manifest, device=device)
    if warm_checkpoint is not None:
        source = load_checkpoint(warm_checkpoint, device=device, evaluation=True)
        warm_start(source.network, model.network)
        manifest['warm_source_graph_sha256'] = source.network.graph.identity()
    return model
