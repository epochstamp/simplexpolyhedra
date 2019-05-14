import sys
import math
import gym
import numpy as np
import scipy
import time
from sklearn.ensemble import ExtraTreesRegressor
import mpu.ml
from simpolyhedra import SimPolyhedra
import polyhedronutils as poly
from copy import deepcopy

"""
        Fitted-Q-Iteration Agent (FQI) with Extra Trees. For continuous state space and discrete state space.
        See http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
"""

def int_to_onehot(i, n):
    return mpu.ml.indices2one_hot([i], nb_classes=n)[0]

class FQI_Agent(object):


    def __init__(self, env, d_prob=2.0):
        self.env = env
        self.RC = ExtraTreesRegressor(n_estimators=200, n_jobs=3)
        self.LS = None
        self.d_prob = d_prob

    def generateRandomTuples(self, N, steps):
        LS = []
        env = self.env
        envs = [deepcopy(env) for _ in range(N)]
        states = [e.reset(poly.randomCubeBasis(e.n//2)) for e in envs]
        range_N = range(N)
        range_steps = range(steps)
        act_histories = [[] for _ in range_N]
        success = 0
        dantzig_trajectory = [True for _ in range_N]
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
                    if act != envs[i].dantzigAction():
                        dantzig_trajectory[i] = False  
                    if done:
                        if r == 1:
                            success += 1
                        envs[i] = None
            
            
            s = ns
        print("Number of trajectories : ", N*steps)
        print("Number of successful trajectories : ", success)
        print("Number of full dantzig trajectories : ", np.sum(dantzig_trajectory))
        return LS




    def toLearningSet(self, LT, i):
        self.env.reset(poly.randomCubeBasis(self.env.n//2))
        if self.LS is None:
            self.LS = np.asarray(list(map(lambda x : np.hstack([x[0], int_to_onehot(x[1], env.getNumberOfActions()), x[3],  [x[2]], [x[4]]]).tolist(),LT)))
        # On first iteration, output is the reward
        if i == 0:
            return self.LS[:,:self.env.getStateSize() + self.env.getNumberOfActions()], self.LS[:,-1]
        # Otherwise, output is r + gamma*max_a Q(s,a)
        # Integer action is converted to one-hot vector.
        inp = self.LS[:,:self.env.getStateSize() + self.env.getNumberOfActions()]
        inp_next = self.LS[:,self.env.getStateSize() + env.getNumberOfActions():2*self.env.getStateSize() + env.getNumberOfActions()]
        
        Q_matrix = np.ones((inp.shape[0],env.getNumberOfActions()))
        j = 0
        for a in range(env.getNumberOfActions()):
            onehot_a = int_to_onehot(a, env.getNumberOfActions())
            tiles = np.tile(onehot_a,(inp.shape[0],1))
            inp_temp = np.hstack([inp_next, tiles])
            out_temp = self.RC.predict(inp_temp)
            Q_matrix[:, j] *= out_temp
            j += 1    
        maxQ_vector = np.amax(Q_matrix, axis=1)
        notdone = 1 - self.LS[:,-1]
        out = self.LS[:,-2] + env.gamma() * maxQ_vector * notdone
        return (inp, out)

    def train(self, I):
        L = self.generateRandomTuples(100, 250)
        print("FQI training")
        for i in range(I):
            print("Iteration ",i,"/",I)
            print("Prepare learning set")
            inp, out = agt.toLearningSet(L, i)
            print("Iterate on extra trees")
            self.RC.fit(inp,out)
        

    def test(self):
        state = self.env.reset(poly.randomCubeBasis(self.env.n//2))
        done = False
        n_actions = env.getNumberOfActions()
        i = 0
        self.RC.n_jobs = 1
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
    print("Training process...")
    agt.train(50)
    print("Training done. Performing test...")
    agt.test()
