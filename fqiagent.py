import sys
import math
import numpy as np
import scipy
import time
import random
import os
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from simpolyhedra import SimPolyhedra
import polyhedronutils as poly
from copy import deepcopy
from joblib import dump, load
import multiprocessing
import argparse
import glob
from shutil import copyfile
import configparser

"""
        Fitted-Q-Iteration Agent (FQI) with Extra Trees. For continuous state space and discrete state space.
        See http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
"""

class SimulationExistsError(Exception):
    pass

class IterIndexSupToMaxIter(Exception):
    pass

class ConfigParseError(Exception):
    pass


class FQI_Agent(object):


    def __init__(self, env, args):
        kwargs = vars(args)
        self.env = env
        self.RC = kwargs["estimator"][0]
        
        self.LS = None
        self.d_prob = kwargs["bias_exploration_coeff"]
        self.cartesian_SA = None
        self.mode = kwargs["feature_mode"]
        
        self.overwrite_mode = kwargs["overwrite_mode"]
        self.args = args
        self.output_folder = kwargs["output_folder"] if kwargs["output_folder"] != "auto" else self._auto_output_folder()
        self.locked = set()
        self.lst_parallel_rpolicy = None

        self.envs_test = self.args.envs_tests

        try:
            self.RC.n_jobs = min(self.RC.n_jobs, self.args.max_njobs)
        except:
            self.RC.n_jobs = 1
        
        try:
            os.makedirs(self.output_folder)    
        except FileExistsError:
            if self.overwrite_mode == "s":
                raise SimulationExistsError("Simulation folder does exists. Erase it or change overwrite_mode")
        if not os.path.isfile(self.output_folder+"/training_parameters.csv") or (self.overwrite_mode == "w" and self.output_folder+"/training_parameters.csv" not in self.locked):
            f = open(self.output_folder+"/training_parameters.csv", "w+")
            header, data = "", ""
            keys = kwargs.keys()
            kwargs["estimator"] = kwargs["estimator"][1][0]+(("#"+"#".join(sorted([k+"="+str(v) for k,v in kwargs["estimator"][1][1].items()]))) if len(kwargs["estimator"][1][1].keys()) > 0 else "") 
            kwargs["geotype"] = self.env.getType()
            kwargs["envs_tests"] = ",".join(kwargs["envs_tests"].keys())
            for k in keys:
                header += k + ";"
                data += str(kwargs[k]) + ";"
            header = header[:-1]
            data = data[:-1]
            header += "\n"
            f.write(header)
            f.write(data)
            f.close()
            self.locked.add(self.output_folder+"/training_parameters.csv")
        

    def _auto_output_folder(self):
        estimator_foldername = "estimator=" + self.args.estimator[1][0] + ("#" if len(self.args.estimator[1][1].keys()) > 0 else "") +  "#".join(sorted([k+"="+str(v) for k,v in self.args.estimator[1][1].items()]))
        return "results/geotype="+self.env.getType()+"/vertices="+str(self.args.vertices)+"/feature_mode="+str(self.args.feature_mode)+"/"+estimator_foldername+"/bias_exploration_coeff="+str(self.d_prob)+"/n_episodes="+str(self.args.n_episodes)+"/horizon_time="+str(self.args.horizon_time)+"/seed="+str(self.args.seed)+"/"

    def generateEpisode(self, tup):
        env, steps, policy, return_mode, perform_reset, id_env = tup
        LS = []
        if perform_reset:
            env.reset()
        done = False
        i = 0
        while not done and i < steps:
            act = policy(env)
            if return_mode == 0:
                features = env.features(act, mode=self.mode)
                _, r, done, _ = env.step(act) 
                LS.append((features,r,done, [env.features(a,mode=self.mode) for a in list(set(env.getAvailableActions()))], i+1))
            elif return_mode == 1:
                _, r, done, _ = env.step(act)
                LS.append((r,i+1))
            i += 1
        return (LS, id_env, done) 

    def generateRandomTuples(self, N, steps):

        if os.path.isfile(self.output_folder+"/learning_set.dmp") and self.overwrite_mode == "a":
            return load(self.output_folder+"/learning_set.dmp")

        if (self.args.max_njobs > 1): 
            with multiprocessing.Pool(self.args.max_njobs) as p:
                LS = p.map(self.generateEpisode, [(deepcopy(self.env), steps, self._randomBiasedPolicy, 0, True, "") for _ in range(N)])
        else:
            LS = [self.generateEpisode((deepcopy(self.env), steps, self._randomBiasedPolicy, 0, True, "")) for _ in range(N)]
        dump(LS, self.output_folder+"/learning_set.dmp", compress=9)

        if not os.path.isfile(self.output_folder+"/training_stats.csv") or (self.overwrite_mode == "w" and self.output_folder+"/training_stats.csv" not in self.locked):
            f = open(self.output_folder+"/training_stats.csv", "w+")
            f.write("n_transitions;n_successes;feature_size;mean_length_success_traj;var_length_success_traj\n")
            n_transitions = N*steps
            n_successes = sum([1 if x[2] else 0 for x in LS])
            length_successes = [len(x[0]) for x in LS if x[2]]
            mean_length_successes = np.mean(length_successes) if len(length_successes) > 0 else -np.inf
            var_length_successes = np.var(length_successes) if len(length_successes) > 0 else 0
            feature_size = self.env.getFeatureSize(mode=self.mode)
            f.write(str(n_transitions) + ";" + str(n_successes) + ";" + str(feature_size) + ";" + str(mean_length_successes) + ";" + str(var_length_successes) + "\n")
            f.close()
            self.locked.add(self.output_folder+"/training_stats.csv")
            
        return LS




    def toLearningSet(self, LT, i):
        if os.path.isfile(self.output_folder+"/learning_fqiset.dmp") and self.overwrite_mode == "a":
            self.LS = load(self.output_folder+"/learning_fqiset.dmp")
            if os.path.isfile(self.output_folder+"/learning_fqinset.dmp"):
                self.LSN = load(self.output_folder+"/learning_fqinset.dmp")
            else:
                LTT = [x[0] for x in LT]
                LS = []
                [LS.extend(e) for e in LTT] 
                self.LSN = np.vstack(map(lambda x : np.vstack(x[-2]), LS))
        self.env.reset()
        if self.LS is None:
            
            LTT = [x[0] for x in LT]
            LS = []
            [LS.extend(e) for e in LTT]
            self.LS = np.asarray(list(map(lambda x : np.hstack([x[0], [x[1]], [x[2]]]).tolist(),LS)))
            self.LSN = np.vstack(map(lambda x : np.vstack(x[3]), LS))
            previous_k = 0
            self.intervals = []
            for _,_,_,lf,_ in LS:
                len_lf = len(lf)
                self.intervals.append((previous_k, previous_k+len_lf))
                previous_k += len_lf
               
        # On first iteration, output is the reward
        if i == 0:
            return self.LS[:,:self.env.getFeatureSize(mode=self.mode)], self.LS[:,-2]
        # Otherwise, output is r + gamma*max_a Q(s,a)
        # Integer action is converted to one-hot vector.

        #old_njobs = self.RC.n_jobs 
        #self.RC.n_jobs = 1
        out_temp = self.RC.predict(self.LSN)
        #self.RC.n_jobs = old_njobs
        Q_values = []

        for (beg,end) in self.intervals:
            Q_values.append(np.max(out_temp[beg:end]))
        maxQ_vector = np.asarray(Q_values)  
        notdone = 1 - self.LS[:,-1]
        inp = self.LS[:,:self.env.getFeatureSize(mode=self.mode)]
        out = self.LS[:,-2] + env.gamma() * maxQ_vector * notdone
           
        return (inp, out)

    def write_log_error(self, error):
        f = open(self.output_folder+"/errors.log","a+")
        f.write(error + "\n")
        f.close()
        
      

    def save_checkpoint(self, i, tested):
        try:
            os.makedirs(self.output_folder+"/checkpoints/") 
        except:
            if (self.overwrite_mode == "w" and self.output_folder+"/*.dmp*" not in self.locked):
                filelist = glob.glob(os.path.join(self.output_folder, "*.dmp*"))
                for f in filelist:
                    os.remove(f)
                self.locked.add(self.output_folder+"/*.dmp*")
        dump({"estimator":self.RC, "iter":i, "tested":tested,"random_state":np.random.get_state()},self.output_folder+"/checkpoints/checkpoint_"+("untested" if not tested else "tested")+".dmp", compress=9)

    def load_checkpoint(self, try_backup = True):
        if (self.overwrite_mode == "w" and self.output_folder+"/*.loaddmp" not in self.locked): 
            self.locked.add(self.output_folder+"/*.loaddmp")
            return None

        #First try to load the tested one
        L = None
        
        try:
            L = load(self.output_folder+"/checkpoints/checkpoint_tested.dmp")
        except:
            self.write_log_error("Unable to load the tested version. Load (try to) the untested one...\n")
            try:
                L = load(self.output_folder+"/checkpoints/checkpoint_tested.dmp")
            except:
                self.write_log_error("Unable to even load the untested version.\n")
        self.write_log_error("\n\n\n******************************************\n\n\n")

        #Try to load backup if not successful above
        if L is None and try_backup:
            if os.path.isfile(self.output_folder+"/checkpoints/checkpoint.dmp.bak"):
                copyfile(self.output_folder+"/checkpoints/checkpoint.dmp.bak", self.output_folder+"/checkpoints/checkpoint.dmp")
            if os.path.isfile(self.output_folder+"/checkpoints/checkpoint_tested.dmp.bak"):
                copyfile(self.output_folder+"/checkpoints/checkpoint_tested.dmp.bak", self.output_folder+"/checkpoints/checkpoint_tested.dmp")
            L = self.load_checkpoint(try_backup=False)
        return L

    def make_backup(self):
        if os.path.isfile(self.output_folder+"/checkpoints/checkpoint.dmp"):
            copyfile(self.output_folder+"/checkpoints/checkpoint.dmp", self.output_folder+"/checkpoints/checkpoint.dmp.bak")
        if os.path.isfile(self.output_folder+"/checkpoints/checkpoint_tested.dmp"):
            copyfile(self.output_folder+"/checkpoints/checkpoint_tested.dmp", self.output_folder+"/checkpoints/checkpoint_tested.dmp.bak")   

    def loop_train_test(self, I):
        L = self.generateRandomTuples(self.args.n_episodes, self.args.horizon_time)
        data = self.load_checkpoint()
        if data is not None:
            numpy.random.set_state(data["random_state"]) 
            beg = data["iter"]
            if beg > I:
                raise IterIndexSupToMaxIter("Set your number of iterations lower than the maximum number (or raise the latter)")
            self.RC = data["estimator"]
            if not data["tested"]:
                self.test()
            
        else:
            beg = 0
        
        start_time = time.time()
        for i in range(beg, I):
            t_elapsed = (time.time() - start_time)/3600 
            if (t_elapsed >= self.args.maximum_time_exec_hours - 0.5):
                print("Simulation interrupted after being executed for " + str(t_elapsed) + " hours.")
                exit(-1) 
            inp, out = agt.toLearningSet(L, i)
            self.RC.fit(inp,out)
            self.make_backup()
            self.save_checkpoint(i, False)
            self.test()
            self.save_checkpoint(i, True)
            

        
    """
    TODO : 
        -> Write tests statistics
        -> Checkpoint/overwrite mode to test
        -> Display : success rate, mean 'diffperf', var 'diffperf', max 'diffperf', min 'diffperf', variable importance. 
    """
    def _randomBiasedPolicy(self,env):
        available_acts = list(set(env.getAvailableActions()))
        probs = [1.0/len(available_acts)] * len(available_acts)
        probs[available_acts.index(env.reflexAction())] = self.d_prob
        probs = list(map(lambda x : x / sum(probs), probs))
        return np.random.choice(available_acts, p=probs)

    def _reflexPolicy(self,env):
        return env.reflexAction()

    def _agentPolicy(self,env):
        available_acts = env.getAvailableActions()
        lst_pred = np.vstack([env.features(a, mode=self.mode) for a in available_acts])
        values = self.RC.predict(lst_pred)
        return available_acts[np.argmax(values)]


    def _writeStatistics(self, LT, f_stats):


        LT_reflex = LT[:len(LT)//2]
        LT_agent = LT[len(LT)//2:]
        for k,f in f_stats.items():
            filtered_stats_agent = [t for t in LT_agent if t[1][0] == k]
            filtered_stats_reflex = [t for t in LT_reflex if t[1][0] == k]
            success_rate = len([t for t in filtered_stats_agent if t[2]]) / len(LT_agent)
            diffperfs = [filtered_stats_agent[i][0][-1][1] - filtered_stats_reflex[i][0][-1][1] for i in range(len(filtered_stats_agent)) if filtered_stats_agent[i][-1]] 
            cond_diffperf = len(diffperfs) > 0
            mean_diffperf = np.mean(diffperfs) if cond_diffperf else -np.inf
            var_diffperf = np.var(diffperfs) if cond_diffperf else 0
            max_diffperf = np.max(diffperfs) if cond_diffperf else -np.inf
            min_diffperf = np.min(diffperfs) if cond_diffperf else -np.inf
            lst_write = [str(success_rate), str(mean_diffperf), str(var_diffperf),str(min_diffperf), str(max_diffperf)]
            f.write(";".join(lst_write) + "\n")
            

    def test(self):
        old_njobs = self.RC.n_jobs
        self.RC.n_jobs = 1
        #Initialize test file if it does not exists
        d_stats = {}

        

        #Write test statistics
        for keyword,testenv in self.envs_test.items():
            if not os.path.isfile(self.output_folder+"/"+keyword+"_stats.csv") or (self.overwrite_mode == "w" and self.output_folder+"/"+keyword+"_stats.csv" not in self.locked):
                f = open(self.output_folder+"/"+keyword+"_stats.csv", "w+")
                f.write("success_rate;mean_diffperf;var_diffperf;min_diffperf;max_diffperf\n")
                
                f.close()
                self.locked.add(self.output_folder+"/"+keyword+"_stats.csv")
            d_stats[keyword] = open(self.output_folder+"/"+keyword+"_stats.csv", "a+")
        if self.lst_parallel_rpolicy is None:
            self.lst_parallel_rpolicy = []
            self.lst_parallel_apolicy = []
            for k,e in self.envs_test.items():
                self.lst_parallel_rpolicy.extend([(deepcopy(e), e.maxSteps, self._reflexPolicy, 1, False, (k,"rpolicy")) for _ in range(self.args.n_episodes_test)])
            [x[0].reset() for x in self.lst_parallel_rpolicy]
            self.lst_parallel_apolicy = [(deepcopy(x[0]), x[1], self._agentPolicy, x[3], x[4], (x[5][0],"apolicy")) for x in self.lst_parallel_rpolicy]

        if self.args.max_njobs > 1:
            with multiprocessing.Pool(self.args.max_njobs) as p:
                LT = p.map(self.generateEpisode, self.lst_parallel_rpolicy + self.lst_parallel_apolicy)
        else:
            LT = [self.generateEpisode(x) for x in self.lst_parallel_rpolicy + self.lst_parallel_apolicy]
        self._writeStatistics(LT, d_stats)

        #Write feature importance
        if not os.path.isfile(self.output_folder+"/feature_importances.csv") or (self.overwrite_mode == "w" and self.output_folder+"/feature_importances.csv" not in self.locked):
            f = open(self.output_folder+"/feature_importances.csv", "w+")
            s = ""
            for i in range(self.RC.feature_importances_.shape[0]):
                s += "feature_importance_"+str(i+1)+";"
            s = s[:-1]
            f.write(s + "\n")
            f.close()
            self.locked.add(self.output_folder+"/feature_importances.csv")
        f = open(self.output_folder+"/feature_importances.csv", "a+")
        f.write(";".join([str(imp) for imp in self.RC.feature_importances_]) + "\n")
        f.close()

        self.RC.n_jobs = old_njobs
        [f.close() for f in d_stats.values()]
        
 

def strictpos_float(astring):    
    has_matched = float(astring) > 0
    if not has_matched:
        raise TypeError("Is supposed to be strictly positive")
    return float(astring)  

def strictpos_int(astring):    
    has_matched = int(astring) > 0
    if not has_matched:
        raise TypeError("Is supposed to be strictly positive")
    return int(astring)

def pos_int(astring):    
    has_matched = int(astring) >= 0
    if not has_matched:
        raise TypeError("Is supposed to be strictly positive")
    return int(astring)      

geometrics = dict()
geometrics["unitcube"] = SimPolyhedra.cube
geometrics["spindle"] = SimPolyhedra.randomSpindle


def geometrical(astring):
    if astring in geometrics:
        return geometrics[astring]
    raise TypeError("Is not a recognized geometrics. List of recognized geometrics : " + list(geometrics.keys()))

estimators = dict()
estimators["extratrees"] = RandomForestRegressor
estimators["randomforest"] = ExtraTreesRegressor
estimators["gdboosting"] = GradientBoostingRegressor

def check_int(s):
    s = str(s)
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit() 

def check_float(s):
    result = False
    if s.count(".") == 1:
        if s.replace(".", "").isdigit():
            result = True
    return result

def autoconvert_type(x):
    if check_int(x): return int(x)
    if check_float(x): return float(x)
    if x.lower() == "true" or x.lower() == "false":
        return x.lower() == "true"
    return x

def estimator(astring):
    sequence = astring.split("#")
    estimator = sequence[0]
    if estimator not in estimators.keys():
        raise TypeError("Is not a recognized estimator. List of recognized estimators : " + list(estimators.keys()))
    args_splitted = [x.split("=") for x in sequence[1:]]
    estimator_args = {x[0]:autoconvert_type(x[1]) for x in args_splitted}
    try:
        return estimators[estimator](**estimator_args), (estimator, estimator_args)
    except:
        s = "Error occured during the instantiation of the estimator. Check that 1) the provided parameters match with the parameters of the model and 2) the preconditions of your parameters are met"
        s += "(e.g., learning_rate does not exists in extratrees, number of estimators cannot be a float in any bagging ensemble..."
        raise TypeError(s)

def configfile(astring):
    if not os.path.isfile(astring):
        raise FileNotFoundError("Configuration file "+configfile+" not found.")
    try:
        config = configparser.ConfigParser()
        config.read(astring)
    except Exception as e:
        raise ConfigParseError("Trouble when opening configuration file. Check for format")
  
    if config.sections() == []:
        print("Warning : Not environment test found. Means no statistics at the end. Up to you...")

    envs = {}
    for section in config.sections(): 
        geotype = config[section]["type"]
        vertices = int(config[section]["vertices"])
        envs[section] = geometrical(geotype)(vertices)
        envs[section].maxSteps = int(config[section]["maxSteps"])
    

    return envs
    

if __name__=="__main__":
    parser = argparse.ArgumentParser("FQI Pivot Agent training for Simplex Environment")
    parser.add_argument("--vertices","-n",help="Number of vertices",type=strictpos_int, default=50)
    parser.add_argument("--geotype","-g",help="Geometrical type of the polyhedron",type=geometrical, default="unitcube")
    parser.add_argument("--seed","-s",help="Random seed",type=int, default=24)
    parser.add_argument("--overwrite-mode", "-O",help="Overwrite mode (w = full overwrite, a = use checkpoint if any (otherwise acts as w), s = don't do anything",choices=["w","a","s"], default="s")
    parser.add_argument("--max-njobs","-j",help="Maximum number of jobs allowed for any parallelizable operation",type=strictpos_int, default=1)
    parser.add_argument("--estimator","-m",help="Estimator (ensemble of trees, neural net...), followed with his own arguments (separated by #)",type=estimator, default="extratrees")
    parser.add_argument("--bias-exploration-coeff","-b",help="Maximal estimator tree depth",type=strictpos_float, default=2.0)
    parser.add_argument("--n-episodes","-E",help="Number of episodes (train)",type=strictpos_int, default=100)
    parser.add_argument("--n-episodes-test","-e",help="Number of episodes per environment test",type=strictpos_int, default=100)
    parser.add_argument("--horizon-time","-T",help="Maximum horizon time before cutting episode",type=strictpos_int, default=100)
    parser.add_argument("--output-folder","-o",help="Output result folder", default="auto")
    parser.add_argument("--feature-mode","-f",help="Featurization mode (see environment specs)", type=pos_int, default=0)
    parser.add_argument("--envs-tests","-F",help="Configuration file for environment testing", type=configfile, default="unitcube.cfg")
    parser.add_argument("--q-iterations","-q",help="Number of iterations of FQI", type=strictpos_int, default=20)
    parser.add_argument("--maximum-time-exec-hours","-l",help="Execution time limit", type=strictpos_int, default=24)
    
    args = parser.parse_args()
    if (args.seed != -1):
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    

    env = args.geotype(args.vertices)
    try:
        agt = FQI_Agent(env, args)
    except SimulationExistsError as e:
        print(e)
        exit(-1)
    agt.loop_train_test(args.q_iterations)
    
