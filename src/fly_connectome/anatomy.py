"""Offline graph selection and anatomy-only acceptance; never consumes Pong metrics."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _indices(graph, body_ids):
    if not np.isin(body_ids, graph.body_ids).all():
        raise ValueError("requested body outside graph")
    return np.searchsorted(graph.body_ids, body_ids)


def _distances(graph, starts, reverse=False, max_hops=None):
    distance = np.full(len(graph.body_ids), -1, dtype=np.int64)
    distance[_indices(graph, starts)] = 0
    pre, post = (graph.post, graph.pre) if reverse else (graph.pre, graph.post)
    for hop in range(1, (max_hops if max_hops is not None else len(distance)) + 1):
        targets = np.unique(post[distance[pre] == hop - 1])
        targets = targets[distance[targets] < 0]
        if not len(targets):
            break
        distance[targets] = hop
    return distance


def _components(graph):
    matrix = csr_matrix((np.ones(len(graph.pre)), (graph.pre, graph.post)),
                        shape=(len(graph.body_ids), len(graph.body_ids)))
    _, labels = connected_components(matrix, directed=True, connection='strong')
    return labels


def select_roster(graph, seeds, motors, *, max_hops, reciprocal_contacts, scc_cap):
    if max_hops < 1 or reciprocal_contacts < 1 or scc_cap < 1:
        raise ValueError("selection limits must be positive")
    forward = _distances(graph, seeds, max_hops=max_hops)
    backward = _distances(graph, motors, reverse=True, max_hops=max_hops)
    keep = (forward >= 0) & (backward >= 0) & (forward + backward <= max_hops)
    keep[_indices(graph, list(seeds) + list(motors))] = True
    # One preregistered reciprocal expansion, using contacts to/from the seed/path set.
    incoming = np.bincount(graph.post, weights=graph.contacts * keep[graph.pre], minlength=len(keep))
    outgoing = np.bincount(graph.pre, weights=graph.contacts * keep[graph.post], minlength=len(keep))
    keep |= (incoming >= reciprocal_contacts) & (outgoing >= reciprocal_contacts)
    labels = _components(graph)
    sizes = np.bincount(labels)
    touched = np.unique(labels[keep])
    keep |= np.isin(labels, touched[sizes[touched] <= scc_cap])
    return graph.body_ids[keep].tolist()


def anatomy_report(graph, sensory_ids, motor_ids, regions):
    if len(regions) != len(graph.body_ids):
        raise ValueError("one region label required per neuron")
    reach = _distances(graph, sensory_ids) >= 0
    active = np.zeros(len(graph.body_ids), dtype=bool)
    active[graph.pre] = True
    active[graph.post] = True
    labels = _components(graph)
    components = []
    sizes = np.bincount(labels)
    recurrent = np.union1d(np.flatnonzero(sizes > 1), labels[graph.pre[graph.pre == graph.post]])
    for label in recurrent:
        members = np.flatnonzero(labels == label)
        components.append(graph.body_ids[members].tolist())
    active_regions = set()
    for i in np.flatnonzero(active):
        labels_for_neuron = regions[i]
        active_regions.update([labels_for_neuron] if isinstance(labels_for_neuron, str) else labels_for_neuron)
    return dict(neurons=len(graph.body_ids), edges=len(graph.pre), contacts=int(graph.contacts.sum()),
                reachable_motor_ids=graph.body_ids[_indices(graph, motor_ids)[reach[_indices(graph, motor_ids)]]].tolist(),
                active_regions=sorted(active_regions),
                recurrent_components=sorted(components), isolated_body_ids=graph.body_ids[~active].tolist())


def choose_threshold(graph, sensory_ids, motor_ids, regions, required_regions):
    if not sensory_ids or not motor_ids:
        raise ValueError("preregistered circuit needs sensory and motor populations")
    for threshold in (5, 3):
        report = anatomy_report(graph.threshold(threshold), sensory_ids, motor_ids, regions)
        if (set(report['reachable_motor_ids']) == set(motor_ids)
                and set(required_regions) <= set(report['active_regions'])):
            return threshold
    raise ValueError("T5 and T3 both break the preregistered circuit")
