"""Explicit source bridge for owned snapshots across reviewed control-only work.

No visual code/model change is admitted. A caller must pin the old snapshot bytes,
its exact ancestor source and the complete reviewed control-file change map.
The copied visual content, calibration/RNG and learning count stay identical.
"""
import copy
import hashlib
import json
import pickle
from pathlib import Path
from .codec import fingerprint
from .online_vision import VisionSnapshot,_copy_visual_model
from .session import source_signature,MODEL_HASH,VISION_CONFIGURATION

TRANSFER_VERSION='owned-vision-control-source-bridge-v1'


def bridge_snapshot(path,*,expected_file_sha256,ancestor_source,reviewed_control_changes):
    path=Path(path);payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_file_sha256:
        raise ValueError('ancestor visual snapshot bytes changed')
    snapshot=pickle.loads(payload)  # Trusted local artifact; hashes are not authentication.
    current=source_signature()
    changes={name:dict(before=ancestor_source.get(name),after=current.get(name))
             for name in sorted(set(ancestor_source)|set(current))
             if ancestor_source.get(name)!=current.get(name)}
    if (not changes or changes!=reviewed_control_changes
            or any(not name.startswith('control/') for name in changes)):
        raise ValueError('exact reviewed control-only source bridge required')
    if (type(snapshot) is not VisionSnapshot or snapshot.source!=ancestor_source
            or snapshot.prior_model_hash!=MODEL_HASH or snapshot.vision_configuration!=VISION_CONFIGURATION
            or type(snapshot.learning_observations) is not int or snapshot.learning_observations<0
            or fingerprint(snapshot.vision)!=snapshot.content_hash):
        raise ValueError('ancestor snapshot model/content/provenance changed')
    owned=_copy_visual_model(snapshot.vision)
    if fingerprint(owned)!=snapshot.content_hash:
        raise ValueError('visual-source bridge changed retained content')
    result=VisionSnapshot(owned,snapshot.content_hash,copy.deepcopy(current),MODEL_HASH,
                          VISION_CONFIGURATION,snapshot.learning_observations).validate()
    receipt=dict(version=TRANSFER_VERSION,ancestor_file_sha256=expected_file_sha256,
        ancestor_source=copy.deepcopy(ancestor_source),new_source=copy.deepcopy(current),
        reviewed_control_changes=copy.deepcopy(changes),visual_content_hash=snapshot.content_hash,
        prior_model_hash=MODEL_HASH,vision_configuration=VISION_CONFIGURATION,
        cumulative_learning_observations=snapshot.learning_observations,
        visual_code_unchanged=True,visual_content_unchanged=True,observations_added=0)
    # Receipt is plain JSON provenance; it never becomes a policy feature.
    json.dumps(receipt,sort_keys=True,allow_nan=False)
    return result,receipt
