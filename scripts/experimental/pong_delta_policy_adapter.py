"""Inference-only adapter for the frozen learned visual delta planner."""
from pong_learned_delta_planner import DeltaPlannerModel, DeltaPlannerController

CONTRACT = 'learned-delta-H8-W8-primitive-replan-realhistory16-actualhit-gate-v1'
load_model = DeltaPlannerModel.load
new_controller = DeltaPlannerController


def choose(controller, scene, baseline_action, baseline_q):
    return controller.choose(scene, baseline_action)


def observe(controller, record):
    controller.observe(record['hits'], record['terminated'])
