import sys
import math
import numpy as np
import scipy
import time
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from simpolyhedra import SimPolyhedra
import polyhedronutils as poly
from copy import deepcopy
from joblib import dump
import multiprocessing

"""
        Fitted-Q-Iteration Agent (FQI) with Extra Trees. For continuous state space and discrete state space.
        See http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
"""


class FQI_Agent(object):


    def __init__(self, env, d_prob=2.0, feature_mode=1):
        self.env = env
        self.RC = ExtraTreesRegressor(max_features=0.75, n_estimators=1000, n_jobs=12)
        self.LS = None
        self.d_prob = d_prob
        self.cartesian_SA = None
        self.mode = feature_mode

    def generateEpisode(self, tup):
        env, steps = tup
        LS = []
        env.reset()
        done = False
        i = 0
        while not done and i < steps:
            available_acts = list(set(env.getAvailableActions()))
            probs = [1.0/len(available_acts)] * len(available_acts)
            probs[available_acts.index(env.dantzigAction())] = self.d_prob
            probs = list(map(lambda x : x / sum(probs), probs))
            act = np.random.choice(available_acts, p=probs)
            features = env.features(act, mode=self.mode)
            _, r, done, _ = env.step(act) 
            LS.append((features,r,done, [env.features(a,mode=self.mode) for a in list(set(env.getAvailableActions()))]))
            i += 1
        return LS 

    def generateRandomTuples(self, N, steps):
        """
        LS = []
        env = self.env
        env.reset()
        envs = [deepcopy(env) for _ in range(N)]
        states = [e.reset() for e in envs]
        range_N = range(N)
        range_steps = range(steps)
        act_histories = [[] for _ in range_N]
        success = 0
        sizes = [0 for _ in range(N)]
        
        for _ in range_steps:
            
            for i in range_N:
                if envs[i] is not None:
                    t = time.time()
                    available_acts = list(set(envs[i].getAvailableActions()))
                    probs = [1.0/len(available_acts)] * len(available_acts)
                    probs[available_acts.index(envs[i].dantzigAction())] = self.d_prob
                    probs = list(map(lambda x : x / sum(probs), probs))
                    act = np.random.choice(available_acts, p=probs)
                    features = envs[i].features(act)
                    ns, r, done, _ = envs[i].step(act) 
                    LS.append((features,r,ns,done, [envs[i].features(a) for a in available_acts]))                
                    states[i] = ns
                    sizes[i] += 1
                    if done:
                        if r == 1:
                            success += 1
                        envs[i] = None
                    print("time elapsed :", time.time() - t)
            
            s = ns
        for i in range(len(sizes)):
            sizes[i] = sizes[i] if envs[i] is None else np.inf
        """
        with multiprocessing.Pool(12) as p:
            LT = p.map(self.generateEpisode, [(deepcopy(self.env), steps) for _ in range(N)])
            LS = []
            [LS.extend(el) for el in LT]
            #LS = [t for sublist in LS for l in sublist for t in l]
        print("Number of transitions : ", N*steps)
        print("Number of successful trajectories : ", sum([1 if x[2] else 0 for x in LS]))
        print("Proportion of 0s : ", sum([sum([(1 if y == 0 else 0) for y in x[0]]) for x in LS]) / (N*steps*self.env.getFeatureSize(mode=self.mode)))
        print("Proportion of -1s : ", sum([sum([(1 if y == -1 else 0) for y in x[0]]) for x in LS]) / (N*steps*self.env.getFeatureSize(mode=self.mode)))
        #print("Mean length of successful trajectories :", np.mean([s for s in sizes if s < np.inf]))
        #print("Mean var length of successful trajectories :", np.var([s for s in sizes if s < np.inf]))
        print("Feature size :", self.env.getFeatureSize(mode=self.mode))
        return LS




    def toLearningSet(self, LT, i):
        self.env.reset()
        if self.LS is None:
            #self.mapact = {a:env.postprocess_action(a, mode=1) for a in range(self.env.getNumberOfActions())}
            self.LS = np.asarray(list(map(lambda x : np.hstack([x[0], [x[1]], [x[2]]]).tolist(),LT)))
            self.LSN = np.vstack(map(lambda x : np.vstack(x[-1]), LT))
            previous_k = 0
            self.intervals = []
            for _,_,_,lf in LT:
                len_lf = len(lf)
                self.intervals.append((previous_k, previous_k+len_lf))
                previous_k += len_lf
               
        # On first iteration, output is the reward
        if i == 0:
            return self.LS[:,:self.env.getFeatureSize(mode=self.mode)], self.LS[:,-2]
        # Otherwise, output is r + gamma*max_a Q(s,a)
        # Integer action is converted to one-hot vector.
        """
        if self.cartesian_SA is None:
            matrix = None 
            intervals = [] 
            j = 1
            k = 0
            submatrices = []
            print("Building cartesian product SxA")
            for _,_,_,ns,_,_,converted_acts,_ in LT:
                pp_acts = np.vstack(converted_acts)
                
                inpnext_tile = np.tile(ns,(pp_acts.shape[0],1))
                submatrices.append(np.hstack([inpnext_tile, pp_acts]))
                if (j-1) % 250 == 0:
                    matrix = np.vstack([matrix] + submatrices) if matrix is not None else np.vstack(submatrices)
                    submatrices = []
                
                
                len_acts = len(converted_acts)
                intervals.append((k,k+len_acts))
                j += 1
                k += len_acts
            print("Cartesian product SxA done")
            self.cartesian_SA = np.vstack([matrix] + submatrices)
            self.intervals = intervals
        """

        old_njobs = self.RC.n_jobs 
        #self.RC.n_jobs = 1
        print("Go predict")
        out_temp = self.RC.predict(self.LSN)
        print("End predict")
        self.RC.n_jobs = old_njobs
        Q_values = []

        for (beg,end) in self.intervals:
            Q_values.append(np.max(out_temp[beg:end]))
        maxQ_vector = np.asarray(Q_values)  
        notdone = 1 - self.LS[:,-1]
        inp = self.LS[:,:self.env.getFeatureSize(mode=self.mode)]
        out = self.LS[:,-2] + env.gamma() * maxQ_vector * notdone
           
             
        """
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
        """
        return (inp, out)

    def train(self, I):
        L = self.generateRandomTuples(1500, 100)
        print("FQI training")
        for i in range(I):
            print("Iteration ",i+1,"/",I)
            print("Prepare learning set")
            inp, out = agt.toLearningSet(L, i)
            print("Iterate on regressor")
            self.RC.fit(inp,out)
        

    def test(self):
        success_rate = 0
        K = 10
        N = 150
        regret_lst = []
        print("Test on ", K, " random basis, limit of ",N," steps : ")
        for _ in range(K):
            self.env.reset()
            env_2 = deepcopy(self.env)
            done = False
            #state = self.env.reset(initial_state)
            i = 0

            #Firstly compute the optimal path
            while not done and i < N:
                _, _, done, _ = self.env.step(self.env.dantzigAction())
                i += 1
            optimal_steps = i
            print("True optimal path is ",optimal_steps," steps !")

            self.RC.n_jobs = 1
            done = False
            i = 0
            while not done and i < N:
                available_acts = env_2.getAvailableActions()
                lst_pred = [env_2.features(a, mode=self.mode) for a in available_acts]
                values = self.RC.predict(lst_pred)
                act = available_acts[np.argmax(values)]
                _, _, done, _ = env_2.step(act)
                i += 1
            if not done:
                print("Optimal solution not found :(")
            else:
                policy_steps = i
                print("Optimal solution found in ", policy_steps, " steps with trees policy !")
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
    dump(agt.RC,"trees.dmp")
    agt.test() 
    
