"""Native survival SARSA with the existing plastic visual learner and new head.

Optional task session only: no launch, teacher, privileged policy features or
change to the Pong path. The final time-limit image bootstraps before reset.
"""
import copy
from dataclasses import asdict
import numpy as np
import torch
from fly_connectome.vision import load_default,VisualStateEncoder
from .codec import fingerprint
from .online_vision import (PersistentVisionSession,VisionSnapshot,MUTABLE_VISION_CONTRACT,
                            observer_tensors,memory_counts)
from .pendulum import PendulumCamera,PendulumPhysics
from .projection import FixedPolicyProjection
from .sarsa import TrueOnlineSarsa
from .scaling import make_policy_transform
from .session import LearningSession,source_signature,model_bytes,MODEL_HASH,VISION_CONFIGURATION


class PendulumLearningSession(LearningSession):
    """Reuse the audited per-camera SARSA/update/reset path, with plastic vision.

    The native environment supplies only pixels, elapsed survival reward and raw
    end flags to this loop. Episode reset retains learned vision and clears scene
    state; no visual snapshot is selected from task reward.
    """
    durable_kind='pendulum-learning-v1'
    __getstate__=PersistentVisionSession.__getstate__
    __setstate__=PersistentVisionSession.__setstate__
    retained_vision=PersistentVisionSession.retained_vision

    def __init__(self,seed=0,*,max_episode_ticks=600,physics=PendulumPhysics(),
                 alpha=.001,lam=.995,gamma_per_second=.95,epsilon=.2,
                 policy_projection=None,policy_scaling=None,vision_snapshot=None):
        if not 0<gamma_per_second<1:
            raise ValueError('continuing survival requires an explicit discount below one per second')
        if policy_projection is not None and policy_scaling is not None:
            raise ValueError('fixed projection owns its post-projection cap')
        model_bytes();physics.validate()
        self.config=dict(seed=seed,max_episode_ticks=max_episode_ticks,physics=asdict(physics),
            alpha=alpha,lam=lam,gamma_per_second=gamma_per_second,epsilon=epsilon,
            reward='elapsed survival seconds; failure terminal; time limit truncation',visual_learning=True)
        self.model_hash,self.vision_configuration=MODEL_HASH,VISION_CONFIGURATION
        self.source=source_signature();self.visual_memory_contract=MUTABLE_VISION_CONTRACT
        self.env=PendulumCamera(seed,max_episode_ticks=max_episode_ticks,config=physics)
        self.vision=load_default();observer=fingerprint(observer_tensors(self.vision))
        self.starting_visual_snapshot=None;self.starting_visual_learning_observations=0
        if vision_snapshot is not None:
            if type(vision_snapshot) is not VisionSnapshot:
                raise ValueError('owned source-validated starting visual snapshot required')
            self.vision=vision_snapshot.clone_model()
            if fingerprint(observer_tensors(self.vision))!=observer:
                raise ValueError('fixed generic visual observer changed')
            self.starting_visual_snapshot=vision_snapshot.content_hash
            self.starting_visual_learning_observations=vision_snapshot.learning_observations
            self.vision.reset_state()
        self.encoder=VisualStateEncoder(64,64,.02)
        self.scaler=make_policy_transform(self.encoder.layout,self.encoder.dimension,policy_scaling)
        dimension=self.encoder.dimension
        if policy_scaling is not None:self.config['policy_scaling']=copy.deepcopy(self.scaler.config)
        if policy_projection is not None:
            self.projection=FixedPolicyProjection(dimension,policy_projection)
            self.config['policy_projection']=copy.deepcopy(self.projection.config)
            dimension=self.projection.output_dimension
        self.learner=TrueOnlineSarsa(3,dimension,alpha=alpha,lam=lam,
            gamma_per_second=gamma_per_second,epsilon=epsilon,seed=seed)
        self.previous=torch.zeros(64,64,dtype=torch.bool)
        self.transitions=self.episodes=self.transition_cameras=0;self.initial_cameras=1
        self.physics_seconds=self.episode_survival_seconds=0.;self.completed_rallies=[]
        self.visual_observations=self.visual_learning_observations=0
        self.latest_encoded=self.observe(self.env.render())
        self.learner.start(*self.policy_features(self.latest_encoded));self.capture_rng()

    def observe(self,frame):
        events=torch.stack((self.previous & ~frame,~self.previous & frame)).float()
        self.previous.copy_(frame)
        encoded=self.encoder.encode(self.vision.step(events,frame.float(),learn=True),frame.float())
        self.visual_observations+=1;self.visual_learning_observations+=1
        return encoded

    def step(self):
        episode=self.episodes;before=self.visual_observations
        record=super().step()
        self.episode_survival_seconds+=record['elapsed_seconds']
        record.update(episode=episode,episode_survival_seconds=self.episode_survival_seconds,
            visual_learning=dict(enabled=True,input_observations=before,
                next_observations=self.visual_observations,
                reset_initial_image_added=bool(record['terminated'] or record['truncated']),
                next_memory_counts=memory_counts(self.vision)))
        if record['terminated'] or record['truncated']:
            self.completed_rallies.append(dict(episode=episode,seconds=self.episode_survival_seconds,
                terminated=record['terminated'],truncated=record['truncated']))
            self.episode_survival_seconds=0.
        return record

    def evaluate(self,transitions):
        raise NotImplementedError('Use a separate fresh frozen task evaluator; this session always learns visually')

    @classmethod
    def validate_restored(cls,session):
        if (type(session) is not cls or session.source!=source_signature()
                or session.model_hash!=MODEL_HASH or session.vision_configuration!=VISION_CONFIGURATION
                or session.visual_memory_contract!=MUTABLE_VISION_CONTRACT
                or type(session.env) is not PendulumCamera
                or session.config['physics']!=asdict(session.env.config)
                or session.config['max_episode_ticks']!=session.env.max_episode_ticks
                or session.env.needs_reset or session.initial_cameras!=session.episodes+1
                or session.visual_learning_observations!=session.visual_observations
                or session.visual_observations!=session.transitions+session.initial_cameras
                or session.transition_cameras!=session.transitions
                or len(session.completed_rallies)!=session.episodes):
            raise ValueError('incompatible pendulum/plastic-vision source/configuration/counters')
        session.env.config.validate();model_bytes()
        for name in ('alpha','lam','gamma_per_second','epsilon'):
            if getattr(session.learner,name)!=session.config[name]:
                raise ValueError('survival learner parameters changed')
        if not 0<session.learner.gamma_per_second<1:
            raise ValueError('continuing survival discount must remain below one')
        projection=getattr(session,'projection',None)
        if projection is not None:
            projection.validate_embedded()
            if (session.config['policy_projection']!=projection.config
                    or session.learner.weights.shape!=(3,projection.output_dimension)):
                raise ValueError('fixed survival projection/configuration changed')
        if (session.learner.transitions!=session.transitions or session.learner.pending_action is None
                or not np.isfinite(session.learner.weights).all() or not np.isfinite(session.learner.traces).all()):
            raise ValueError('survival learner pending state/counters are invalid')
        import random
        random.setstate(session.python_rng);np.random.set_state(session.numpy_rng);torch.set_rng_state(session.torch_rng)
        return session
