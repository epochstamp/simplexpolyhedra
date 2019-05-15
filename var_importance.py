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
from joblib import load

        
         
if __name__=="__main__":
    env = SimPolyhedra.cube(50)
    env.reset(poly.cube_randomBasis(env.n//2))
    RC = load("trees.dmp")
    tableau_importance = np.reshape(RC.feature_importances_[:-env.getNumberOfActions()], env.state.shape)
    print("Objective importance", tableau_importance[0,0])
    sum_action_importance = np.sum(RC.feature_importances_[-env.getNumberOfActions():])
    mean_action_importance = np.mean(RC.feature_importances_[-env.getNumberOfActions():])
    var_action_importance = np.var(RC.feature_importances_[-env.getNumberOfActions():])
    print("Action importance moments (sum, mean,var) :", sum_action_importance, mean_action_importance, var_action_importance)
    sum_rcost_importance = np.sum(tableau_importance[0,1:])
    mean_rcost_importance = np.mean(tableau_importance[0,1:])
    var_rcost_importance = np.var(tableau_importance[0,1:])
    print("Reduced cost importance moments (sum,mean,var)", sum_rcost_importance,mean_rcost_importance, var_rcost_importance)
    sum_bvec_importance = np.sum(tableau_importance[1:,0])
    mean_bvec_importance = np.mean(tableau_importance[1:,0])
    var_bvec_importance = np.var(tableau_importance[1:,0])
    print("b vector importance moments (sum,mean,var)", sum_bvec_importance,mean_bvec_importance, var_bvec_importance)
    sum_amat_importance = np.sum(tableau_importance[1:,1:])
    mean_amat_importance = np.mean(tableau_importance[1:,1:])
    var_amat_importance = np.var(tableau_importance[1:,1:])
    print("A matrix importance moments (sum,mean,var)", sum_amat_importance,mean_amat_importance, var_amat_importance)
