"""Public visual-ID input adapter for unchanged learned relational parameters."""
from pong_relational_persistent_binding import PersistentRelationalPolicy,PersistentRelationalController

CONTRACT='persistent-published-public-visual-ids-same-relational-parameters-v1'
load_model=PersistentRelationalPolicy.load
new_controller=PersistentRelationalController


def choose(controller,public_entities,baseline_action,baseline_q):
    return controller.choose(public_entities,baseline_action)


def observe(controller,record):
    controller.observe(record['hits'],record['terminated'])
