import sys
import math
import gym
import numpy as np
import scipy
import time
from sklearn.ensemble import ExtraTreesRegressor

from simpolyhedra import SimPolyhedra
import polyhedronutils as poly
from copy import deepcopy
from joblib import dump

"""
        Fitted-Q-Iteration Agent (FQI) with Extra Trees. For continuous state space and discrete state space.
        See http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
"""


class FQI_Agent(object):


    def __init__(self, env, d_prob=2.0):
        self.env = env
        self.RC = ExtraTreesRegressor(n_estimators=200, n_jobs=4)
        self.LS = None
        self.d_prob = d_prob

    def generateRandomTuples(self, N, steps):
        LS = []
        env = self.env
        envs = [deepcopy(env) for _ in range(N)]
        states = [e.reset(poly.cube_randomBasis(e.n//2)) for e in envs]
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
        self.env.reset(poly.cube_randomBasis(self.env.n//2))
        if self.LS is None:
            self.LS = np.asarray(list(map(lambda x : np.hstack([x[0], env.postprocess_action(x[1], mode=0), x[3],  [x[2]], [x[4]]]).tolist(),LT)))
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
            onehot_a = env.postprocess_action(a, mode=0)
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
        L = self.generateRandomTuples(200, 250)
        print("FQI training")
        for i in range(I):
            print("Iteration ",i,"/",I)
            print("Prepare learning set")
            inp, out = agt.toLearningSet(L, i)
            print("Iterate on extra trees")
            self.RC.fit(inp,out)
        

    def test(self):
        n_actions = env.getNumberOfActions()
        success_rate = 0
        K = 10
        N = 500
        regret_lst = []
        print("Test on ", K, " random basis, limit of ",N," steps : ")
        for _ in range(K):
            done = False
            state = self.env.reset(poly.cube_randomBasis(self.env.n//2))
            i = 0

            #Firstly compute the optimal path
            while not done and i < N:
                _, _, done, _ = self.env.step(self.env.dantzigAction())
                i += 1
            optimal_steps = i
            print("True optimal path is ",optimal_steps," steps !")

            self.RC.n_jobs = 1
            state = self.env.reset(poly.cube_randomBasis(self.env.n//2))
            done = False
            i = 0
            while not done and i < N:
                max_a = -np.inf
                argmax_a = None
                for a in env.getAvailableActions():
                    inp = np.hstack([state, env.postprocess_action(a, 0)])
                    pred = self.RC.predict([inp])[0]
                    if pred > max_a:
                        max_a = pred
                        argmax_a = a
                if max_a == -np.inf:
                    break
                else: state, _, done, _ = self.env.step(argmax_a)
                i += 1
            if not done:
                print("Optimal solution not found :(")
            else:
                policy_steps = i
                print("Optimal solution found in ", i, " steps !")
                success_rate += 1
                regret_lst.append(np.abs(optimal_steps - policy_steps))
        print("Success rate : ",str(float(success_rate/K)))
        print("Mean 'regret' : ", np.mean(regret_lst) if len(regret_lst) > 0 else -np.inf)
        print("Variance 'regret' : ", np.var(regret_lst) if len(regret_lst) > 1 else 0)
        
         
if __name__=="__main__":
    env = SimPolyhedra.cube(50)
    agt = FQI_Agent(env)
    print("Training process...")
    agt.train(75)
    print("Training done. Performing test...")
    agt.test() 
    dump(agt.RC,"trees.dmp")
