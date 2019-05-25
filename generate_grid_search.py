"""
#!/bin/bash
# Submission script for Vega
#SBATCH --job-name=simpolyhedra_fqiagent_%ijob
#SBATCH --time=20:00:00 # hh:mm:ss
#SBATCH --cpus-per-task=12
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6144 # 6GB
#SBATCH --partition=defq


srun python fqiagent.py --n-episodes %nepisodes --horizon-time %horizontime --max-njobs 12 --estimator %estimator --overwrite-mode a --geotype unitcube --vertices 50 --feature-mode %featuremode --bias-exploration-coeff %biasexplorationcoeff --seed %seed --envs-tests unitcube.cfg --n-episodes-test 40 
"""

from itertools import product
import numpy as np
import os
import hashlib

def my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))



def computeMD5hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()

commands = dict()
commands["nepisodes"] = [100,500,1000]
commands["horizontime"] = [100,250]
commands["estimator"] = ["extratrees#n_estimators=100#n_jobs=4",
                         "extratrees#n_estimators=250#n_jobs=6", 
                         "extratrees#n_estimators=500#n_jobs=8", 
                         "extratrees#n_estimators=1000#n_jobs=12", 
                         "randomforest#n_estimators=100#n_jobs=4",
                         "randomforest#n_estimators=250#n_jobs=6", 
                         "randomforest#n_estimators=500#n_jobs=8", 
                         "randomforest#n_estimators=1000#n_jobs=12",
                         "gdboosting#n_estimators=100#loss=huber#warm_start=True",
                         "gdboosting#n_estimators=250#loss=huber#warm_start=True", 
                         "gdboosting#n_estimators=500#loss=huber#warm_start=True",
                         "gdboosting#n_estimators=100#loss=huber",
                         "gdboosting#n_estimators=250#loss=huber", 
                         "gdboosting#n_estimators=500#loss=huber"]
commands["featuremode"] = [0,1,2]
commands["biasexplorationcoeff"] = [1.0,2.0,4.0]
np.random.seed(200)
commands["seed"] = list(set(np.random.choice(10000,30,replace=False)))


if __name__=="__main__":
   patternslurm = open("patternized_slurm.sh").read()
   i = 0
   try:
       os.makedirs("slurm_jobs")
   except Exception as e:
       pass 
   for combination in my_product(commands):
       
       pattern_transform  = str(patternslurm)
       jobname = ""
       for k,v in combination.items():
           pattern_transform = pattern_transform.replace("%"+k, str(v))
           jobname += "k="+str(k)+"_v="+str(v)+"_"
       jobname = jobname[:-1]
       pattern_transform = pattern_transform.replace("%ijob", computeMD5hash(jobname))
       file_to_finalscript = open("slurm_jobs/"+jobname+".sh", "w+")
       file_to_finalscript.write(pattern_transform)
       file_to_finalscript.close()       
       i += 1
       
    
