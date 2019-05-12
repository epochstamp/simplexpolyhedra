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

def int_to_onehot(i, n):
    return mpu.ml.indices2one_hot([i], nb_classes=n)[0]

class FQI_Agent(object):


    def __init__(self, env, d_prob=0.66):
        self.env = env
        self.RC = ExtraTreesRegressor(n_estimators=100)
        self.LS = None
        self.d_prob = d_prob

    def generateRandomTuples(self, N, steps):
        LS = []
        env = self.env
        envs = [deepcopy(env) for _ in range(N)]
        states = [e.reset(-1) for e in envs]
        range_N = range(N)
        range_steps = range(steps)
        act_histories = [[] for _ in range_N]
        success = 0
        for _ in range_steps:
            
            for i in range_N:
                if envs[i] is not None:
                    available_acts = list(set(envs[i].getAvailableActions()))
                    probs = [1.0/len(available_acts)] * len(available_acts)
                    probs[available_acts.index(envs[i].dantzigAction())] = self.d_prob
                    probs = list(map(lambda x : x / sum(probs), probs))
                    act = np.random.choice(available_acts, p=probs)
                    act_histories[i].append(act)
                    ns, r, done, _ = envs[i].step(act) 
                    LS.append((states[i],act,r,ns,done))
                    states[i] = ns  
                    if done:
                        if r == 1:
                            success += 1
                        envs[i] = None
            
            
            s = ns
        print("Number of trajectories : ", N*steps)
        print("Number of successful trajectories : ", success)
        return LS




    def toLearningSet(self, LT, i):
        self.env.reset(-1)
        if self.LS is None:
            self.LS = np.asarray(list(map(lambda x : np.hstack([x[0], int_to_onehot(x[1], env.getNumberOfActions()), x[3],  [x[2]], [x[4]]]).tolist(),LT)))
        # On first iteration, output is the reward
        if i == 0:
            return self.LS[:,:self.env.getStateSize() + self.env.getNumberOfActions()], self.LS[:,-1]
        # Otherwise, output is r + gamma*max_a Q(s,a)
        # It is assumed that RC is a list of regressor, one per action
        # (It does not make sense to include it in features, it is an index)
        inp = self.LS[:,:self.env.getStateSize() + self.env.getNumberOfActions()]
        inp_next = self.LS[:,self.env.getStateSize() + env.getNumberOfActions():-2]
        out = self.LS[:,-2] + env.gamma() * self.RC.predict(inp_next) * self.LS[:,-1]
        return (inp, out)

    def train(self, I):
        L = self.generateRandomTuples(75, 100)
        for i in range(I):
            inp, out = agt.toLearningSet(L, i)
            self.RC.fit(inp,out)

    def test(self):
        state = self.env.reset(-1)
        done = False
        n_actions = env.getNumberOfActions()
        i = 0
        while not done and i < 1000:
            max_a = -np.inf
            argmax_a = None
            for a in env.getAvailableActions():
                inp = np.hstack([state, int_to_onehot(a, n_actions)])
                pred = self.RC.predict([inp])[0]
                if pred > max_a:
                    max_a = pred
                    argmax_a = a
            if max_a == -np.inf:
                done = True
            else: state, reward, done, _ = self.env.step(argmax_a)
            i += 1
        if not done:
            print("Optimal solution not found :(")
        else:
            print("Optimal solution found in ", i, " steps !")
         
if __name__=="__main__":
    env = SimPolyhedra.cube(50)
    agt = FQI_Agent(env)
    agt.train(20)
    agt.test()
