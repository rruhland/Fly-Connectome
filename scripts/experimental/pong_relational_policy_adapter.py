"""Portable parameter payload and actual-feedback adapter for a learned servo."""
import hashlib
import json
from pong_relational_feedback import RelationalPolicy,RelationalController

CONTRACT='anonymous-observed-pair-relational-servo-actual-command-feedback-v1'
load_model=RelationalPolicy.load
new_controller=RelationalController


def from_payload(payload):
    return RelationalPolicy(*payload['choices'],payload['coefficients'],payload['scales'])


def to_payload(model):
    return dict(choices=list(model.choices),coefficients=model.coefficients.tolist(),scales=model.scales.tolist())


def model_hash(payload):
    model=from_payload(payload)
    return hashlib.sha256(json.dumps(to_payload(model),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def choose(controller,scene,baseline_action,baseline_q):
    return controller.choose(scene,baseline_action)


def observe(controller,record):
    controller.observe(record['hits'],record['terminated'])
