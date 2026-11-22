"""Actor-relative motion descriptors and a duration-only value model."""

import numpy as np


FRAMES = 24
EXTENDED = 44
FRACTIONS = np.array((0., .50, .88, 1.25, 1.65, 2.10, 2.65))


def sample_path(paths,ticks):
    base = ticks.astype(int)
    rows = np.arange(len(paths))[:,None]
    weight = (ticks-base)[...,None]
    return (1.-weight)*paths[rows,base]+weight*paths[rows,np.minimum(base+1,paths.shape[1]-1)]


def describe(paths, actor, incoming):
    """Give 2D and 3D paths a common actor-relative scale and approach direction."""
    paths = np.asarray(paths, dtype=float)
    relative = paths[:, 0]-actor
    distance = np.maximum(1.e-7, np.linalg.norm(relative, axis=-1))
    axis = relative/distance[:, None]
    speed = max(1.e-7, np.linalg.norm(incoming))
    toward = np.maximum(.30*speed, -axis@incoming)
    clock = np.clip(distance/toward, 4., 11.)
    ticks = np.minimum(clock[:,None]*FRACTIONS,FRAMES)
    samples = sample_path(paths,ticks)
    relative = (samples-actor)/distance[:, None, None]
    forward = np.einsum('btd,bd->bt',relative,axis)
    transverse = np.linalg.norm(relative-forward[...,None]*axis[:, None], axis=-1)
    delta = np.diff(samples,axis=1)/np.maximum(.8,np.diff(ticks,axis=1))[...,None]
    longitudinal_speed = np.einsum('btd,bd->bt',delta,axis)/speed
    magnitude = np.linalg.norm(delta,axis=-1)/speed
    alignment = np.einsum('btd,d->bt',delta,incoming)/(speed*np.maximum(1.e-7,np.linalg.norm(delta,axis=-1)))
    features = (forward[:,1:],transverse[:,1:],longitudinal_speed,magnitude,alignment)
    if paths.shape[1]>FRAMES+1:
        late = np.minimum(clock[:,None]*np.array((2.65,3.15,3.70,4.35)),paths.shape[1]-1)
        future = (sample_path(paths,late)-actor)/distance[:,None,None]
        outgoing = np.einsum('btd,bd->bt',future,axis)
        features += (outgoing[:,1:],np.diff(outgoing,axis=1),)
    return np.column_stack(features)


def fit(x, y):
    mean, scale = x.mean(axis=0), np.maximum(.04, x.std(axis=0))
    normalized = np.clip((x-mean)/scale,-4.,4.)
    design = np.column_stack((normalized,np.square(normalized)))
    baseline = design.mean(axis=0)
    design -= baseline
    penalty = 11.*np.eye(design.shape[1])
    weights = np.linalg.solve(design.T@design+penalty,design.T@(y-y.mean()))
    return dict(mean=mean.tolist(),scale=scale.tolist(),center=baseline.tolist(),weights=weights.tolist(),
                target_mean=float(y.mean()))


def score(model,x):
    mean,scale = np.array(model['mean']),np.array(model['scale'])
    normalized = np.clip((x-mean)/scale,-4.,4.)
    terms = np.column_stack((normalized,np.square(normalized)))-model['center']
    return terms@model['weights'] + model['target_mean']
