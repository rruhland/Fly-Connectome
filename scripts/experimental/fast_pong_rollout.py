"""Lean research episodes over the unchanged legal Pong evaluation path.

No journals, per-camera hashes, oracle features or learning rule. Research callers
own episode-level accounting/checkpoints. An interrupted process has no arbitrary
camera recovery guarantee: its unacknowledged episode grant must stay charged.
"""
from dataclasses import asdict
import hashlib
import math
import random
import time
import traceback

import numpy as np
import torch

from fly_connectome.control.codec import fingerprint
from fly_connectome.control.fresh_evaluation import preserve_global_rngs
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig

EXECUTION_KIND='lean-pong-episode-v1'


def run_episode(weights,config,*,first_hit_cutoff=True,max_transitions=None,
                deadline_monotonic=None,trace=False,terminal_digest=False,global_seed=904271833):
    """Return exact counted episode outcomes; optional records aid small checks.

    Reuse FreshPongEvaluator itself, including its camera/vision/encoder, K1 greedy
    decision, native transition, post-action observation and first-hit boundary.
    Frozen contents are checked at construction and closure rather than each frame.
    A caller may cooperatively stop on a camera count or wall deadline. Partial
    episodes never constitute complete training fitness. A failed final step is
    reported as ambiguous (including a possible initial image), never hidden.
    """
    if isinstance(config,dict):config=PongEvaluationConfig(**config)
    if type(config) is not PongEvaluationConfig:raise ValueError('PongEvaluationConfig required')
    config.validate()
    if config.episodes!=1 or config.policy_kind!='trained' or config.action_hold_frames!=1:
        raise ValueError('lean research supports one trained greedy K1 episode')
    if type(first_hit_cutoff) is not bool or type(trace) is not bool or type(terminal_digest) is not bool:
        raise ValueError('boolean execution flags required')
    if max_transitions is not None and (type(max_transitions) is not int or max_transitions<0):
        raise ValueError('nonnegative camera grant required')
    if deadline_monotonic is not None and (not isinstance(deadline_monotonic,(int,float)) or not math.isfinite(deadline_monotonic)):
        raise ValueError('finite monotonic deadline required')
    if type(global_seed) is not int or not 0<=global_seed<2**32:raise ValueError('uint32 global bookkeeping seed required')
    began=time.monotonic();records=[] if trace else None;attempted=0;failed=False;error=None;reason=None
    with preserve_global_rngs():
        random.seed(global_seed);np.random.seed(global_seed);torch.manual_seed(global_seed)
        session=FreshPongEvaluator(weights,config,first_hit_cutoff=first_hit_cutoff)
        try:
            while not session.complete:
                if max_transitions is not None and session.transitions>=max_transitions:
                    reason='camera grant';break
                if deadline_monotonic is not None and time.monotonic()>=deadline_monotonic:
                    reason='wall deadline';break
                attempted+=1
                record=session.step(check_frozen=False)
                if trace:records.append(record)
            session.verify_frozen()
        except Exception:
            failed=True;reason='episode failure';error=traceback.format_exc()
        report=session.summary()
        state_hash=fingerprint(session) if terminal_digest and not failed else None
        learned_hash=session.initial_learned_hash
    exact=bool(not failed)
    result=dict(execution_kind=EXECUTION_KIND,config=asdict(config),first_hit_cutoff=first_hit_cutoff,
        complete=bool(report['complete'] and not failed),failed=failed,stop_reason=reason or 'episode complete',
        report=report,transitions=report['transitions'],initial_cameras=report['initial_cameras'],
        attempted_transition_cameras=attempted,observations_exact=exact,
        transition_camera_upper_bound=max(attempted,report['transitions']),
        initial_camera_upper_bound=max(report['initial_cameras'],int(attempted>0)),
        ambiguous_last_step=bool(failed and attempted>report['transitions']),
        frozen_parameters_checked=not failed,learned_hash=learned_hash,terminal_state_hash=state_hash,
        wall_seconds=time.monotonic()-began,error=error)
    if trace:result['records']=records
    return result


def run_batch(weights,configs,**options):
    """Sequential episode worker; scheduler may parallelize batches externally."""
    return [run_episode(weights,config,**options) for config in configs]
