#!/bin/bash
# Submission script for Vega
#SBATCH --job-name=shfq_%ijob
#SBATCH --time=20:00:00 # hh:mm:ss
#SBATCH --cpus-per-task=12
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8192 # 8GB

source activate myenv
srun python fqiagent.py --n-episodes %nepisodes --horizon-time %horizontime --max-njobs 12 --estimator %estimator --overwrite-mode a --geotype %geotype --vertices %vertices --feature-mode %featuremode --bias-exploration-coeff %biasexplorationcoeff --seed %seed --envs-tests unitcube.cfg --n-episodes-test 40 --maximum-time-exec-hours 20 --q-iterations 100
