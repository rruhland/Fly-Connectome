"""Pinned MaleCNS source provenance and explicit neurotransmitter sign resolution."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

BASE_URL = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
RELEASE_FILES = (
    'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'body-neurotransmitters-male-cns-v1.0.feather',
    'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
)


@dataclass(frozen=True)
class Source:
    filename: str
    url: str
    sha256: str


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source(path, source):
    actual = checksum(path)
    if actual != source.sha256:
        raise ValueError(f"checksum mismatch: {source.filename}")
    return actual


def write_manifest(path, sources, body_ids, rules, graph_hash):
    payload = dict(schema_version=1, dataset='male-cns:v1.0',
                   source_page='https://male-cns.janelia.org/download/',
                   license='CC-BY-4.0', sources=[asdict(s) for s in sources],
                   body_ids=list(body_ids), extraction_rules=rules, graph_sha256=graph_hash)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def resolve_signs(body_ids, neurotransmitters, sign_map, minimum_confidence):
    if not 0 <= minimum_confidence <= 1 or not set(sign_map.values()) <= {-1, 1}:
        raise ValueError("invalid sign/confidence policy")
    rows = {row['body']: row for row in neurotransmitters}
    signs, unresolved = [], []
    for body in body_ids:
        row = rows.get(body, {})
        confidence = row.get('predicted_nt_confidence')
        nt = row.get('predicted_nt')
        if confidence is None or not confidence >= minimum_confidence or nt not in sign_map:
            unresolved.append(body)
        else:
            signs.append(sign_map[nt])
    if unresolved:
        raise ValueError(f"unresolved neurotransmitter signs for body IDs: {unresolved}")
    return signs
