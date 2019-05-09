import sys
import math
import gym
import numpy as np
import time


class SimPolyhedra():

    """
    Discount factor
    """
    def gamma(self): return 0.95

    def __init__(self, dim):
        """
        Instantation of a Simplex-Polyhedra problem
        for standardized linear programs (represented
        by polyhedras)
        Parameters :
            - dim (integer) - number of variables of the problem
        """
        self.dim = dim


    def reset(self):
        """
        Sample and return an initial state
        (may be fixed)
        """
        #self.state = something
        return self.observe()
    


    def step(self, act):
        """
        Transition step from state to successor given a discrete action
        Parameters : 
            - act \in [0...dim-1] : index of a variable (tableau column)
        """
        reward = 0
        done = False
        return self.observe(), reward, done, {}
 
    def observe(self): 
        """
        Transform state to observation (identity here)
        """
        return np.array(self.state)

