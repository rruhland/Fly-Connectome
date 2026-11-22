"""Estimate physical trajectory value from nearby archived experience."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

from .tt22_value_calibration import kinetics, matrix


MODES = ('kernel','kernel_shuffled')
RANK = 8
NEIGHBOR = 5
PRIOR_WEIGHT = .30


class SupportedValue:
    def __init__(self,checkpoint,mode):
        _,data,labels = matrix(checkpoint['training'])
        self.labels = labels[checkpoint['shuffle_order']] if mode=='kernel_shuffled' else labels
        self.center = data.mean(axis=0)
        self.scale = np.maximum(data.std(axis=0),.05)
        normalized = (data-self.center)/self.scale
        _,singular,axes = np.linalg.svd(normalized,full_matrices=False)
        self.axes = axes[:RANK].T
        self.axis_scale = np.maximum(singular[:RANK]/np.sqrt(len(data)),1.)
        self.examples = self.encode(data)
        self.weights = np.ones(len(self.examples))
        neighbors = np.partition(cdist(self.examples,self.examples),NEIGHBOR,axis=1)[:,NEIGHBOR]
        self.bandwidth = float(np.median(neighbors))

    def encode(self,data):
        return (((data-self.center)/self.scale)@self.axes)/self.axis_scale

    def predict(self,data):
        distance = cdist(self.encode(data),self.examples,metric='sqeuclidean')
        weight = np.exp(-.5*distance/self.bandwidth**2)*self.weights
        return (weight@self.labels)/(PRIOR_WEIGHT+weight.sum(axis=1))

    def __call__(self,descriptors,paths,actor):
        return self.predict(np.column_stack((descriptors,kinetics(paths,actor))))

    def observe(self,features,reward,weight):
        self.examples = np.vstack((self.examples,self.encode(np.asarray(features))))
        self.labels = np.append(self.labels,reward)
        self.weights = np.append(self.weights,weight)

    def state(self):
        fields = ('center','scale','axes','axis_scale','examples','labels','weights')
        return {**{name:getattr(self,name).tolist() for name in fields},'bandwidth':self.bandwidth}

    @classmethod
    def from_state(cls,state):
        model = cls.__new__(cls)
        for name,value in state.items():
            setattr(model,name,value if name=='bandwidth' else np.asarray(value))
        model.axes = np.asfortranarray(model.axes)
        return model


def quality(models,checkpoint):
    valid = checkpoint['validation']
    _,features,reward = matrix(valid)
    returned = np.array([row['actual_return']>0 for row in valid])
    count = int(sum(returned))
    result = {}
    for name,model in models.items():
        predicted = model.predict(features)
        auc = (rankdata(predicted)[returned].sum()-count*(count+1.)/2.)/(count*(len(valid)-count))
        result[name] = dict(return_auc=float(auc),reward_spearman=float(spearmanr(reward,predicted).statistic),
                            reward_mae=float(np.mean(abs(predicted-reward))),bandwidth=model.bandwidth)
    return result
