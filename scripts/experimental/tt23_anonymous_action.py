"""Learn choice of a constant motor displacement in native Pong from duration alone."""

import argparse

import numpy as np

from .tt3_numeric import NumericMemory
from .tt3_pong import _native_rallies
from .tt3_pong_probe import ShorterDwell
from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_pong import CAP, OPPONENT_SPEED, ROLE_MODEL, OpticalDurationAgent, cases
from .tt23_optical_contact import OUT

OPTIONS=np.linspace(-.27,.27,7)
WIDTH=np.array([.14,.34,.25,.34])
CHECK=OUT/'anonymous-action-source.json.gz'


def features(relative,velocity):
    speed=max(1.e-8,np.linalg.norm(velocity))
    return np.array([relative[0],velocity[0]/speed,abs(relative[1]),np.sign(relative[1])*velocity[1]/speed])


def proposal(model, state):
    x=np.asarray(model['x'])
    values=np.asarray(model['value'])
    dist=np.sum(((x-state)/WIDTH)**2,axis=1)
    weight=np.exp(-.5*np.minimum(100.,dist))
    # The same symmetric prior applies to every distinct motor displacement.
    return (weight@values + .7*values.mean())/(weight.sum()+.7)


class FixedMotor(OpticalDurationAgent):
    def __init__(self, role,seed,option,model=None):
        super().__init__(role,NumericMemory([np.zeros(3)]),seed,option=0,motor_factory=ShorterDwell)
        self.move=option
        self.model=model
        self.context=None
        self.offset=None

    def choose(self):
        if self.motor.axis is None:
            return self.motor.probe()
        if self.actor is None:
            return self.motor.effects[1]
        axis=self.motor.axis
        body=np.asarray(self.actor.position)
        if self.context is None and len(self.mover_history)>=2:
            normal=np.array([-axis[1],axis[0]])
            # Compare with the visual motor reference to which each discrete jog is anchored.
            relative=self.mover_history[-1]-self.motor.ready
            velocity=(self.mover_history[-1]-self.mover_history[0])/(len(self.mover_history)-1)
            self.context=features(np.array([relative@axis,relative@normal]),np.array([velocity@axis,velocity@normal]))
            if self.model is not None:
                self.move=int(np.argmax(proposal(self.model,self.context)))
            self.offset=float(OPTIONS[self.move])
        goal=self.motor.ready if self.offset is None else self.motor.ready+self.offset*axis
        vel=(self.actor_history[-1]-self.actor_history[0])/max(1,len(self.actor_history)-1)
        return self.motor.action(float((goal-body-6.*vel)@axis))


def collect(source_cases,spec,role):
    groups={k:[] for k in spec}
    queue=[(c,k) for c in source_cases for k in spec]
    for begin in range(0,len(queue),42):
        batch=queue[begin:begin+42]
        agents=[FixedMotor(role,c.exploration,*spec[k]) for c,k in batch]
        trials=_native_rallies([c for c,k in batch],agents,max_ticks=CAP,opponent_speed=OPPONENT_SPEED)
        for (c,k),agent,row in zip(batch,agents,trials,strict=True):
            groups[k].append(dict(row,option=agent.move,context=None if agent.context is None else agent.context.tolist()))
    return groups


def train(rows):
    out={'x':[],'value':[]}
    shuffled={'x':[],'value':[]}
    rng=np.random.default_rng(708357637)
    for i in range(len(rows['0'])):
        experiments=[rows[str(j)][i] for j in range(7)]
        state=experiments[0]['context']
        if state is None:
            continue
        if any(r['context']!=state for r in experiments):
            raise ValueError('motors changed the source scene before selection')
        seconds=np.array([r['ticks']/120. for r in experiments])
        duration=np.log1p(seconds)
        duration-=duration.mean()
        noise=rng.permutation(duration)
        for model,values in [(out,duration),(shuffled,noise)]:
            model['x'].extend([state,(-np.array(state)*[1,1,-1,-1]).tolist()])
            model['value'].extend([values.tolist(),values[::-1].tolist()])
    return out,shuffled


def source():
    import torch
    torch.set_num_threads(1)
    role=RoleModel(**_read(ROLE_MODEL))
    source_cases=cases(72,1559000000)
    held=cases(36,1614000000)
    training=collect(source_cases,{str(j):(j,None) for j in range(7)},role)
    model,shuffled=train(training)
    evaluation=collect(held,{'learned':(3,model),'shuffled':(3,shuffled),'neutral':(3,None)},role)
    summary={k:dict(ticks=float(np.mean([r['ticks'] for r in rows])),
                    first_contacts=sum(bool(r['contacts'] and r['contacts'][0]['returned']) for r in rows),
                    identified=sum(r['context'] is not None for r in rows)) for k,rows in evaluation.items()}
    _write(CHECK,dict(role_sha256=_hash(ROLE_MODEL),options=OPTIONS.tolist(),width=WIDTH.tolist(),training=training,
                      conditions=evaluation,model=model,shuffled=shuffled,summary=summary))
    print(summary,flush=True)


if __name__=='__main__':
    source()
