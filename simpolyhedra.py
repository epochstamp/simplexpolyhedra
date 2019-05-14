import sys
import math
#import gym
import numpy as np
from scipy import optimize
import scipy
import time

import polyhedronutils as poly

def feasibleBasis(A,b,method='SimPolyhedra'):
    m = A.shape[0]
    n = A.shape[1]
    A_p = np.hstack([A,np.eye(m)])
    for i in range(m):
        if b[i] < 0:
            A_p[i,n+i] = -1.

    c = np.hstack([np.zeros([1,n]),np.ones([1,m])])
    
    if method == 'SimPolyhedra':
        P = SimPolyhedra(A_p,b,c) 
        P.reset([False]*n + [True]*(m))
        
        while not P.isOptimal():
            a = P.dantzigAction()
            P.step(a)
        
        return P.basis[:n]
        
    elif method == 'scipy':
        result = scipy.optimize.linprog(c,A_ub=A_p,b_ub=b)
        if result.fun == 0:    
            basis = [True if val > 0 else False for val in result.x]
            s = sum(basis)
            i = 0
            while s < m:
                if result.x[i] == 0:
                    basis[i] = True
                    s += 1
                i += 1
            return basis[:n]
        else:
            print("Error : no feasible basis found")
    

class SimPolyhedra():

    """
    Discount factor
    """
    def gamma(self): return 0.95

    def __init__(self, A, b, c):
        """
        Instantiation of a Simplex-Polyhedra problem
        for standardized linear programs (represented
        by polyhedras)
        Parameters :
            - n (integer) - number of variables of the problem
            - m (integer) - number of constraints of the problem
            - A (m x n matrix) - constraint matrix
            - b (m x 1 matrix) - constraint bounds (such that Ax <= b)
            - c (1 x n matrix) - variable costs
        """
        self.m = b.shape[0]
        self.n = c.shape[1]
        
        self.A = A
        self.b = b
        self.c = c

    def getNumberOfActions(self): return self.n

    def cube(n,obj='random'):
        """
        Instantiation of a Simplex-Polyhedra on a
        n-dimensional cube
        Parameters :
            - n (integer) - dimension of the cube
            - obj (str) - 'random' for a random objective
                        - 'pos_random' for a random positive objective
                        - 'unit' for an unit objective (sum of coordinates)
                        - 'negunit' for the opposite unit objective (-sum of coordinates)
        """
        A,b = poly.cube(n)
        c = poly.objective(n,obj)
            
        A,b,c = poly.standardForm(A,b,c)
        
        return SimPolyhedra(A,b,c)
        
    def randomSpindle(n,cond=100,obj='random'):
        """
        Instantiation of a Simplex-Polyhedra on a
        n-dimensional spindle
        Parameters :
            - n (integer) - dimension of the spindle
            - cond (float) - condition number of the spindle
            - obj (str) - 'random' for a random objective
                        - 'pos_random' for a random positive objective
                        - 'unit' for an unit objective (sum of coordinates)
                        - 'neg_unit' for the opposite unit objective (-sum of coordinates)
        """
        A,b = poly.spindle_random(n,cond)
        c = poly.objective(n,obj)
            
        A,b,c = poly.standardForm(A,b,c)
        
        return SimPolyhedra(A,b,c)
        
    def randomPolyhedron(n):
        A = -np.random.random([2*n,n])
        b = np.zeros([2*n,1])
        b[:,0] = np.sum(A,axis=1)/2.
        A = np.vstack([A,np.eye(n)])
        b = np.vstack([b,np.ones([n,1])*100])
        c = poly.objective(n,'pos_random')
            
        A,b,c = poly.standardForm(A,b,c)
        
        return SimPolyhedra(A,b,c)
        
    def reset(self,initBasis=None,randomSteps=0):
        """
        Sample and return an initial state
        (may be fixed)
        """
        if initBasis is None:
            self.basis = feasibleBasis(self.A,self.b)
        else:
            self.basis = initBasis[:]
        
        # These expressions can be seen in Section 1.2
        A_B_inv = np.linalg.inv(self.A[:,self.basis])
        A_r = A_B_inv.dot(self.A)
        b_r = A_B_inv.dot(self.b)
        c_r = self.c - self.c[0,self.basis].dot(A_r)
        z_r = -self.c[:,self.basis].dot(b_r)
        
        # Storage of informations on basic variables
        self.rowVar = []
        for i in range(self.n):
            if self.basis[i]:
                self.rowVar.append(i)
        
        # Creation of the state (containing the entire simplex "tableau" )
        self.state = np.vstack([np.hstack([z_r,c_r]),np.hstack([b_r,A_r])])
        
        for k in range(randomSteps):
            a = np.random.choice(self.getAvailableActions())
            self.step(a)
        
        return self.observe()

    def getStateSize(self):
        return self.observe().flatten().shape[0]

    def getAvailableActions(self):
        return [i for i in range(self.n) if not self.basis[i]]
    
    def getImprovingActions(self):
        return [i for i in range(self.n) if not self.basis[i] and self.state[0,1+i] < 0]
    
    def randomImprovingAction(self):
        return np.random.choice(self.getImprovingActions())
    
    def dantzigAction(self):
        return np.argmin(self.state[0,1:])
        
    def greatestImprovementAction(self):
        max_obj = -np.inf
        argmax_obj = None
        for act in range(self.n):
            e,m = -1,np.inf
            for i in range(self.m):
                if self.state[1+i,1+act] > 0:
                    r = self.state[1+i,0]/self.state[1+i,1+act]
                    if r <= m:
                        m = r
                        e = i
            
            # e == -1 means the polyhedron is not bounded (Section 1.3 step 2)
            assert(e != -1)
            
            obj = self.state[0,0] - self.state[1+e,0]*self.state[0,1+act]/self.state[1+e,1+act]
            if obj > max_obj:
                argmax_obj = act
                max_obj = obj
        
        return argmax_obj
    
    def steepestEdgeAction(self):
        min_c = np.inf
        argmin_c = None
        for act in range(self.n):
            c = self.state[0,1+act]/np.linalg.norm(self.state[1:,1+act])
            if c < min_c:
                min_c = c
                argmin_c = act
        
        return argmin_c
        
    def step(self, act):
        """
        Transition step from state to successor given a discrete action
        Parameters : 
            - act \in [0...dim-1] : index of a variable (tableau column)
        """
        reward = 0
        done = False
        
        # Only nonbasic variables can enter the basis and induce a change
        if not self.basis[act]:
            e,m = -1,np.inf
            for i in range(self.m):
                if self.state[1+i,1+act] > 0:
                    r = self.state[1+i,0]/self.state[1+i,1+act]
                    if r <= m:
                        m = r
                        e = i
            
            # e == -1 means the polyhedron is not bounded (Section 1.3 step 2)
            assert(e != -1)
            
            # Applying computations seen in Section 1.3 step 3
            E = np.eye(self.m+1)
            E[:,1+e] = -self.state[:,1+act]/self.state[1+e,1+act]
            E[1+e,1+e] = 1./self.state[1+e,1+act]
            
            self.state = E.dot(self.state)
            
            # Update of informations on basic variables
            self.basis[self.rowVar[e]] = False
            self.basis[act] = True
            self.rowVar[e] = act
            
            # Termination if optimal
            if self.isOptimal():
                reward = 1
                done = True
                
        return self.observe(), reward, done, {}
 
    def observe(self): 
        """
        Transform state to observation (matrix to vector here)
        """
        return self.state.flatten()

    def objective(self):
        return -self.state[0,0]
    
    def solution(self):
        x = np.zeros([self.n])
        for i in range(self.m):
            x[self.rowVar[i]] = self.state[1+i,0]
        return x
            
    def isOptimal(self):
        return (self.state[0,1:]+0.0001 >= 0).all()
    
    
if __name__ == '__main__':
    n = 50
    P = SimPolyhedra.cube(n)
    """ 
    2 ways to initialize a basis
    automatically (with no input) :
    """
    #P.reset()
    """
    or directly by specifying a basis :
    """
    P.reset(poly.cube_randomBasis(n))
    """
    when you specify a basis, 
    you can also specify a number of random steps. 
    This will generate a random path from the 
    initial basis, giving you a random basis
    """
    #P.reset(poly.cube_randomBasis(n),100)
    
    """ We look at the initial feasible base i.e the initial vertex """
    reference_basis = P.basis[:]
    
    """ Dantzig's rule """
    P.reset(reference_basis)
    steps = 0
    acts = []
    while not P.isOptimal():
        print("Objective = " + str(P.objective()))
        a = P.dantzigAction()
        P.step(a)
        acts.append(a)
        steps += 1
    print("Objective = " + str(P.objective()))
    print("")
    
    expected = sum(P.c[P.c<=0])
    print("Expected objective: " + str(expected))
    print("Objective: " + str(P.objective()))
    print("")
    
    """ For a cube, we know the expected number of steps """
    expected_steps = sum([P.basis[i] != reference_basis[i] for i in range(P.n)])//2
    print("Expected number of steps : " + str(expected_steps))
    print("Number of steps : " + str(steps))
    print("History of actions + uniqueness: ", acts, len(set(acts)) == len(acts))
    print("")
    
    """ Greatest Improve rule """
    P.reset(reference_basis)
    steps = 0
    acts = []
    while not P.isOptimal():
        print("Objective = " + str(P.objective()))
        a = P.greatestImprovementAction()
        P.step(a)
        acts.append(a)
        steps += 1
    print("Objective = " + str(P.objective()))
    print("")
    
    expected = sum(P.c[P.c<=0])
    print("Expected objective: " + str(expected))
    print("Objective: " + str(P.objective()))
    print("")
    
    """ For a cube, we know the expected number of steps """
    expected_steps = sum([P.basis[i] != reference_basis[i] for i in range(P.n)])//2
    print("Expected number of steps : " + str(expected_steps))
    print("Number of steps : " + str(steps))
    print("History of actions + uniqueness: ", acts, len(set(acts)) == len(acts))
    print("")
    
    """ Steepest edge rule """
    P.reset(reference_basis)
    steps = 0
    acts = []
    while not P.isOptimal():
        print("Objective = " + str(P.objective()))
        a = P.steepestEdgeAction()
        P.step(a)
        acts.append(a)
        steps += 1
    print("Objective = " + str(P.objective()))
    print("")
    
    expected = sum(P.c[P.c<=0])
    print("Expected objective: " + str(expected))
    print("Objective: " + str(P.objective()))
    print("")
    
    """ For a cube, we know the expected number of steps """
    expected_steps = sum([P.basis[i] != reference_basis[i] for i in range(P.n)])//2
    print("Expected number of steps : " + str(expected_steps))
    print("Number of steps : " + str(steps))
    print("History of actions + uniqueness: ", acts, len(set(acts)) == len(acts))
    print("")