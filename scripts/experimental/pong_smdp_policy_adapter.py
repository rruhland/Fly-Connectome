"""Small inference adapter for the fixed SMDP forest and actual macro contract."""
from pong_smdp_adaptive_forest import AdaptiveForest,SMDPController

CONTRACT='smdp16-or-actualhit-naturalend-baseline1-publiccamera-mod5-v1'
load_model=AdaptiveForest.load
new_controller=SMDPController


def choose(controller,scene,baseline_action,baseline_q):
    return controller.choose(scene,baseline_action)


def observe(controller,record):
    controller.observe(record['hits'],record['terminated'])
