"""Persistent existing visual learning with a frozen policy and exact journaling.

This optional session does not change the visual representation or policy rule.
It never launches an experiment. Ordinary frozen evaluators keep their contract.
"""
import copy
from dataclasses import dataclass
import torch
from .codec import fingerprint
from .environment import PongCamera
from .fresh_evaluation import preserve_global_rngs
from .pong_evaluation import FreshPongEvaluator
from .session import MODEL_HASH,VISION_CONFIGURATION,source_signature

MUTABLE_VISION_CONTRACT='observer-pinned-mutable-visual-memory-v1'


def observer_tensors(vision):
    return {'observer.weights':vision.observer.weights,'observer.bias':vision.observer.bias}


def learned_visual_state(vision):
    """The same learned-bank/calibration fields frozen by ordinary evaluation."""
    return (vision.state.memory.keys,vision.state.memory.values,vision.state.memory.metric,
            vision.dynamics,vision.calibration)


def memory_counts(vision):
    return dict(context_keys=int(vision.state.memory.keys.shape[0]),
        dynamics={str(h):{branch:int(getattr(model,branch).keys.shape[0]) for branch in ('short','long')}
                  for h,model in sorted(vision.dynamics.items())})


def _context_cache_validity(vision):
    fields=vars(vision.state.memory)
    if '_variance_version' not in fields:return None
    return fields['_variance_version']==fields['_variance_keys']._version


def _restore_context_cache_validity(vision,valid):
    """Retain the existing digest's logical validity, not a local Torch counter."""
    memory=vision.state.memory
    if valid is None:
        if '_variance_version' in vars(memory):raise ValueError('unexpected context cache')
    elif type(valid) is bool and '_variance_keys' in vars(memory):
        memory._variance_version=memory._variance_keys._version if valid else -1
    else:raise ValueError('invalid serialized context cache validity')


def _copy_visual_model(vision):
    valid=_context_cache_validity(vision)
    cloned=copy.deepcopy(vision)
    _restore_context_cache_validity(cloned,valid)
    return cloned


@dataclass(frozen=True)
class VisionSnapshot:
    """Owned canonical scene-reset model; caller mutations fail content validation."""
    vision: object
    content_hash: str
    source: dict
    prior_model_hash: str
    vision_configuration: str
    learning_observations: int

    def __getstate__(self):
        return dict(vars(self)),_context_cache_validity(self.vision)

    def __setstate__(self,state):
        fields,valid=state
        for name,value in fields.items():object.__setattr__(self,name,value)
        _restore_context_cache_validity(self.vision,valid)

    def validate(self):
        if (self.source!=source_signature() or self.prior_model_hash!=MODEL_HASH
                or self.vision_configuration!=VISION_CONFIGURATION
                or type(self.learning_observations) is not int or self.learning_observations<0
                or fingerprint(self.vision)!=self.content_hash):
            raise ValueError('retained visual snapshot source/content changed')
        return self

    def clone_model(self):
        self.validate()
        cloned=_copy_visual_model(self.vision)
        if fingerprint(cloned)!=self.content_hash:raise ValueError('retained snapshot clone changed content')
        return cloned


class PersistentVisionSession(FreshPongEvaluator):
    """One visual model across episodes; readout/scaler/observer/config stay pinned.

    learn_enabled=False is a fresh frozen-snapshot probe. True is prequential
    control: each actual initial/transition image updates existing visual memory.
    Explicit episode seeds start new physics/camera phases without replacing vision;
    omitting them preserves the original evaluator's continuing decision cadence.
    Initial and final images both count as actual visual observations. Scene resets
    clear observer traces, tracker state, endpoint context keys, histories and pending
    forecast credits, while retaining context memory, all dynamics banks/private RNGs
    and branch/context/endpoint calibration. A snapshot copies the entire model,
    then applies exactly that same scene reset; it never shares tensor storage with
    the live stream. No policy update or fitted-policy export occurs here.
    """
    durable_kind='pong-persistent-vision-v1'

    def __getstate__(self):
        return dict(vars(self)),_context_cache_validity(self.vision)

    def __setstate__(self,state):
        fields,valid=state
        self.__dict__.update(fields)
        _restore_context_cache_validity(self.vision,valid)

    def __init__(self,weights,config,*,learn_enabled,episode_seeds=None,vision_snapshot=None):
        if (type(learn_enabled) is not bool or config.policy_kind!='trained'
                or config.skip_baseline_perception or config.blank_camera or config.action_hold_frames!=1):
            raise ValueError('persistent vision requires trained hold1 live-camera control and a bool learning mode')
        if episode_seeds is not None:
            episode_seeds=tuple(episode_seeds)
            if (len(episode_seeds)!=config.episodes or not episode_seeds or episode_seeds[0]!=config.seed
                    or any(type(s) is not int or s<0 for s in episode_seeds)):
                raise ValueError('explicit episode seed schedule must match config and initial serve')
        self.visual_memory_contract=MUTABLE_VISION_CONTRACT
        self.learn_enabled=learn_enabled
        self.episode_seeds=episode_seeds
        self.visual_observations=self.visual_learning_observations=0
        self.starting_visual_snapshot=None
        self.starting_visual_learning_observations=0
        super().__init__(weights,config)
        default_observer_hash=fingerprint(observer_tensors(self.vision))
        with preserve_global_rngs():
            if vision_snapshot is not None:
                if type(vision_snapshot) is not VisionSnapshot:raise ValueError('owned validated VisionSnapshot required')
                self.vision=vision_snapshot.clone_model()
                if fingerprint(observer_tensors(self.vision))!=default_observer_hash:
                    raise ValueError('retained snapshot changed the fixed observer')
                self.vision.reset_state()
                self.starting_visual_snapshot=vision_snapshot.content_hash
                self.starting_visual_learning_observations=vision_snapshot.learning_observations
            self.initial_learned_hash=self.learned_hash()
        self.capture_rng()

    def learned_hash(self):
        fixed=(self.policy.weights,observer_tensors(self.vision),self.scaler,self.config,self.env.game.config,
               self.visual_memory_contract,self.learn_enabled,self.episode_seeds,self.starting_visual_snapshot,
               self.starting_visual_learning_observations)
        if not self.learn_enabled:fixed+=(learned_visual_state(self.vision),)
        return fingerprint(fixed)

    def _observe(self,frame):
        events=torch.stack((self.previous & ~frame,~self.previous & frame)).float()
        self.previous.copy_(frame)
        self.latest_encoded=self.encoder.encode(
            self.vision.step(events,frame.float(),learn=self.learn_enabled),frame.float())
        self.visual_observations+=1
        self.visual_learning_observations+=int(self.learn_enabled)

    def _begin_episode(self):
        if self.episode_seeds is not None and self.episodes>0:
            self.env=PongCamera(self.episode_seeds[self.episodes],
                max_episode_ticks=self.config.max_episode_ticks,opponent_speed=self.config.opponent_speed)
        # Existing reset_state preserves learned banks; do not load_default here.
        super()._begin_episode()

    def step(self):
        if self.complete:raise StopIteration('persistent visual stream complete')
        if self.learned_hash()!=self.initial_learned_hash:
            raise RuntimeError('persistent vision changed pinned policy, observer, mode or frozen probe memory')
        with preserve_global_rngs():
            if self.initial_cameras==self.episodes:self._begin_episode()
        consumed=dict(observations=self.visual_observations,learning_observations=self.visual_learning_observations,
                      memory_counts=memory_counts(self.vision))
        record=super().step()
        record['visual_learning']=dict(enabled=self.learn_enabled,input=consumed,
            next_observations=self.visual_observations,next_learning_observations=self.visual_learning_observations,
            next_memory_counts=memory_counts(self.vision))
        return record

    def retained_vision(self):
        """Own the complete model including RNG/seen/capacity/calibration state.

        Scene-local unfinished credits are deliberately discarded at this explicit
        transfer boundary; a live journal checkpoint instead retains every field.
        No extra observations or learning occur while creating the snapshot.
        """
        with preserve_global_rngs():
            model=_copy_visual_model(self.vision);model.reset_state()
        return VisionSnapshot(model,fingerprint(model),copy.deepcopy(self.source),MODEL_HASH,
                              VISION_CONFIGURATION,self.starting_visual_learning_observations+self.visual_learning_observations)

    def summary(self):
        report=super().summary()
        report.update(visual_learning_enabled=self.learn_enabled,visual_memory_contract=self.visual_memory_contract,
            visual_observations=self.visual_observations,visual_learning_observations=self.visual_learning_observations,
            starting_visual_learning_observations=self.starting_visual_learning_observations,
            cumulative_visual_learning_observations=self.starting_visual_learning_observations+self.visual_learning_observations,
            visual_memory_counts=memory_counts(self.vision),learned_visual_state_hash=fingerprint(learned_visual_state(self.vision)),
            starting_visual_snapshot=self.starting_visual_snapshot,
            prior_model_hash=MODEL_HASH,episode_seeds=None if self.episode_seeds is None else list(self.episode_seeds))
        return report

    @classmethod
    def validate_restored(cls,session):
        if (type(session) is not cls or session.visual_memory_contract!=MUTABLE_VISION_CONTRACT
                or type(session.learn_enabled) is not bool or session.config.policy_kind!='trained'
                or session.config.skip_baseline_perception or session.config.blank_camera
                or session.config.action_hold_frames!=1 or hasattr(session,'camera_intervention')
                or hasattr(session,'first_hit_cutoff')
                or type(session.visual_observations) is not int or session.visual_observations<0
                or type(session.visual_learning_observations) is not int
                or session.visual_learning_observations!=(session.visual_observations if session.learn_enabled else 0)
                or type(session.starting_visual_learning_observations) is not int
                or session.starting_visual_learning_observations<0
                or (session.starting_visual_snapshot is None and session.starting_visual_learning_observations!=0)
                or session.visual_observations!=session.transition_cameras+session.initial_cameras):
            raise ValueError('invalid persistent visual-learning session contract/counters')
        seeds=session.episode_seeds
        if seeds is not None and (type(seeds) is not tuple or len(seeds)!=session.config.episodes
                or seeds[0]!=session.config.seed or any(type(s) is not int or s<0 for s in seeds)):
            raise ValueError('invalid restored persistent episode schedule')
        return super().validate_restored(session)
