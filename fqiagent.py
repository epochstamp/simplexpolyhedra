import sys
import math
import gym
import numpy as np
import scipy
import time
from sklearn.ensemble import ExtraTreesRegressor

"""
	Fitted-Q-Iteration Agent (FQI) with Extra Trees. For continuous state space and discrete state space.
        See http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
"""

def FQI_Agent():


    def __init__(self, I, env):
        self.I = I
        self.env = env
        self.RC = []
        for i in range(env.getNumberOfActions()):
            self.RC.append(ExtraTreesRegressor(n_estimators=100))

    def generateRandomTuples(self, N):
        LS = []
        env = self.env
        for _ in range(N):
            s = env.reset()
            a = np.random.randint(low=0,high=env.getNumberOfActions())
            ns, r, _, _ = env.step(a)
            LS.append((s,a,r,ns))
        return LT


    def toLearningSet(self, LT, i):
        # On first iteration, output is the reward
        if i == 0:
            return list(map(lambda x : x[:-1],LS))
        # Otherwise, output is r + gamma*max_a Q(s,a)
        # It is assumed that RC is a list of regressor, one per action
        # (It does not make sense to include it in features, it is an index)
        
         
			
