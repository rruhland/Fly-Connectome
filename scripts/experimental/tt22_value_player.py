"""Use nominal physical trajectories with a transferred value as the sole ranker."""

from hashlib import sha256

from mujoco import rollout
import numpy as np

from .tt4_arena import BALL_INERTIA, BALL_MASS, MAGNUS_COEFFICIENT, SPIN_DECAY, mujoco
from .tt19_factory import RATE
from .tt19_model_rollout import ORIGIN
from .tt21_body_state import StateInitializedDrySimulator
from .tt22_rally import RallyPlayer
from .tt22_value import FRAMES, describe, score


LIMIT = 112


class TrajectoryDrySimulator(StateInitializedDrySimulator):
    """Advance the nominal robot and mover, exposing positions without scoring events."""

    def paths(self,tapes,lengths,observation,horizon):
        state = self.twin.state()
        batch = len(tapes)
        current = np.broadcast_to(state.physics,(batch,len(state.physics))).copy()
        current[:,self.ball_p:self.ball_p+3] = observation.center+ORIGIN
        current[:,self.ball_p+3:self.ball_p+7] = (1.,0.,0.,0.)
        current[:,self.ball_v:self.ball_v+3] = observation.velocity
        current[:,self.ball_v+3:self.ball_v+6] = observation.spin
        position = np.empty((batch,horizon+1,3))
        position[:,0] = current[:,self.ball_p:self.ball_p+3]-ORIGIN
        drive = np.broadcast_to(state.targets,(batch,self.n)).copy()
        progress = np.zeros((batch,len(self.axes)))
        control = np.zeros((batch,self.timestep,self.n+6*self.model.nbody))
        ticks = np.arange(1,self.timestep+1)[None,:,None]
        half = self.timestep//2
        for frame in range(horizon):
            requested = tapes[:,min(frame,tapes.shape[1]-1)]
            rates = np.where((frame<lengths)[:,None],requested,-np.clip(progress,-1.,1.))
            progress += rates
            wanted = drive[:,None,self.axes]+ticks*(rates*self.increments)[:,None]
            settings = np.clip(wanted,self.low,self.high)
            control[:,:,:self.n] = state.targets
            control[:,:,self.axes] = settings
            drive[:,self.axes] = settings[:,-1]
            for start in (0,half):
                q = current[:,self.ball_p+3:self.ball_p+7]
                motion = current[:,self.ball_v:self.ball_v+3]
                local = current[:,self.ball_v+3:self.ball_v+6]
                axis = np.cross(q[:,1:],local)
                twist = local+2.*(q[:,:1]*axis+np.cross(q[:,1:],axis))
                force = BALL_MASS*(-self.drag*np.linalg.norm(motion,axis=1,keepdims=True)*motion
                                   + MAGNUS_COEFFICIENT*np.cross(twist,motion))
                control[:,start:start+half,self.force_id:self.force_id+3] = force[:,None]
                control[:,start:start+half,self.force_id+3:self.force_id+6] = (-BALL_INERTIA*SPIN_DECAY*twist)[:,None]
                portion = np.ascontiguousarray(control[:,start:start+half])
                predicted,_ = rollout.rollout(self.model,self.data,current,portion,control_spec=self.controls)
                current = np.ascontiguousarray(predicted[:,-1])
            position[:,frame+1] = current[:,self.ball_p:self.ball_p+3]-ORIGIN
        return position


class ValuePlayer(RallyPlayer):
    """Rank every feasible timed proposal by optical future duration value."""

    def __init__(self,law,model,gains,horizon=FRAMES):
        super().__init__(law)
        self.sim.close()
        self.sim = TrajectoryDrySimulator(law)
        self.value_model = model
        self.gains = np.asarray(gains)
        self.horizon = horizon
        self.reset()

    def reset(self):
        super().reset()
        self.actor_reading = None
        self.actor_origin = None
        self.transfer_proposals = False

    def lateral_contact(self,estimate,center):
        if not self.transfer_proposals:
            return center[0]
        start = self.actor_origin[0]
        seen = estimate[0][0]
        displacement,motion = self.gains
        return start + displacement*(seen-start) + motion*(center[0]-seen)

    def candidate_tapes(self,estimate,target):
        self.transfer_proposals = False
        default,placements = self.goal_tapes(estimate,target)
        self.transfer_proposals = True
        shifted,relative = self.goal_tapes(estimate,target)
        self.transfer_proposals = False
        tapes = np.concatenate((default,shifted))
        params = [dict(p,proposal='geometric') for p in placements]+[dict(p,proposal='portable') for p in relative]
        if len(tapes)>LIMIT:
            selected = np.linspace(0,len(tapes)-1,LIMIT).astype(int)
            return tapes[selected],[params[index] for index in selected]
        return tapes,params

    def search(self,estimate,observation,target):
        if self.actor_reading is None:
            model,data = self.sim.model,self.sim.twin.data
            mujoco.mj_kinematics(model,data)
            self.actor_origin = data.geom('learner_racket').xpos.copy()-ORIGIN
        else:
            self.actor_origin = np.asarray(self.actor_reading)
        tapes,params = self.candidate_tapes(estimate,target)
        if not len(tapes):
            return None
        lengths = np.clip(np.ceil([p['eta'] for p in params]).astype(int)+3,3,tapes.shape[1])
        paths = self.sim.paths(tapes,lengths,observation,self.horizon)
        features = describe(paths,self.actor_origin,observation.velocity/30.)
        if callable(self.value_model):
            values = self.value_model(features,paths,self.actor_origin)
        else:
            values = score(self.value_model,features)
        index = int(np.argmax(values))
        digest = sha256(np.ascontiguousarray(tapes).tobytes()+lengths.tobytes()).hexdigest()
        log = dict(params[index],candidate_count=len(tapes),candidate_sha256=digest,choice=index,
                   predicted_value=float(values[index]),features=features[index].tolist(),
                   actor_origin=self.actor_origin.tolist(),path=paths[index].tolist())
        return tapes[index,:lengths[index]],log
