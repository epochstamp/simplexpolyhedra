import sys
import math
import gym
import numpy as np
import scipy
import time
from sklearn.ensemble import ExtraTreesRegressor
import mpu.ml
from simpolyhedra import SimPolyhedra
from copy import deepcopy

"""
        Fitted-Q-Iteration Agent (FQI) with Extra Trees. For continuous state space and discrete state space.
        See http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
"""

class FQI_Agent(object):


    def __init__(self, I, env):
        self.I = I
        self.env = env
        self.RC = ExtraTreesRegressor(n_estimators=100)
        self.LS = None

    def generateRandomTuples(self, N, steps):
        LS = []
        env = self.env
        envs = [deepcopy(env) for _ in range(N)]
        states = [e.reset(-1) for e in envs]
        range_N = range(N)
        range_steps = range(steps)
        for _ in range_steps:
            acts = np.random.randint(low=0,high=env.getNumberOfActions(),size=N)
            for i in range_N:
                if envs[i] is not None:
                    ns, r, done, _ = envs[i].step(acts[i]) 
                    LS.append((states[i],acts[i],r,ns,done))
                    states[i] = ns  
                    if done:
                        print("yay")
                        envs[i] = None
            
            
            s = ns
        return LS


    def toLearningSet(self, LT, i):

        if self.LS is None:
            LS = list(map(lambda x : np.hstack([x[0], mpu.ml.indices2one_hot([x[1]], nb_classes=env.getNumberOfActions())[0], [x[2]]]).tolist(),LT))
        # On first iteration, output is the reward
        if i == 0:
            return LS
        # Otherwise, output is r + gamma*max_a Q(s,a)
        # It is assumed that RC is a list of regressor, one per action
        # (It does not make sense to include it in features, it is an index)
        
         
if __name__=="__main__":
    env = SimPolyhedra.cube(15)
    agt = FQI_Agent(10, env)
    L = agt.generateRandomTuples(100, 1000)
    LS = agt.toLearningSet(L, 0)
    print(len(LS[0]))
