import sys
import math
try:
    import gym
except:
    pass
import numpy as np
from scipy import optimize
import scipy
import time
try:
    import mpu.ml
except:
    pass
import polyhedronutils as poly

def sign(x,tol=1e-8):
    if abs(x) < tol:
        return 0
    elif x > 0:
        return 1
    else:
        return -1

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
            print(P.objective())
        
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

    def initFeatures(self,tol = 1e-8):
        self.staticFeatures = np.zeros([27,self.n])
        
        c_pos = np.sum(abs(self.c[self.c > 0]))
        c_neg = np.sum(abs(self.c[self.c < 0]))
        
        A_pos = np.zeros([self.m,1])
        A_neg = np.zeros([self.m,1])
        for j in range(self.m):
            A_pos[j,0] = np.sum(self.A[j,self.A[j,:]>0])
            A_neg[j,0] = -np.sum(self.A[j,self.A[j,:]<0])
        
        for i in range(self.n):
            # cost function features
            self.staticFeatures[0,i] = sign(self.c[0,i])
            self.staticFeatures[1,i] = abs(self.c[0,i])/c_pos
            self.staticFeatures[2,i] = abs(self.c[0,i])/c_neg
            
            # constraint coefficient features
            ppA = self.A[self.A[:,i]>0,i]/A_pos
            pnA = self.A[self.A[:,i]>0,i]/A_neg
            npA = self.A[self.A[:,i]<0,i]/A_pos
            nnA = self.A[self.A[:,i]<0,i]/A_neg
            
            self.staticFeatures[3,i] = np.min(ppA) if ppA.shape[1] > 0 else -1
            self.staticFeatures[4,i] = np.max(ppA) if ppA.shape[1] > 0 else -1
            self.staticFeatures[5,i] = np.max(pnA) if pnA.shape[1] > 0 else -1
            self.staticFeatures[6,i] = np.max(pnA) if pnA.shape[1] > 0 else -1
            self.staticFeatures[7,i] = -np.min(npA) if npA.shape[1] > 0 else -1
            self.staticFeatures[8,i] = -np.max(npA) if npA.shape[1] > 0 else -1
            self.staticFeatures[9,i] = -np.max(nnA) if nnA.shape[1] > 0 else -1
            self.staticFeatures[10,i] = -np.max(nnA) if nnA.shape[1] > 0 else -1
            
            # bounds/constraint features
            pbA = self.A[self.b[:,0]>0,i]/self.b[self.b[:,0]>0,0]
            nbA = self.A[self.b[:,0]<0,i]/(-self.b[self.b[:,0]<0,0])
            self.staticFeatures[11,i] = abs(np.min(pbA)) if pbA.shape[0] > 0 else -1
            self.staticFeatures[12,i] = sign(np.min(pbA)) if pbA.shape[0] > 0 else 0
            self.staticFeatures[13,i] = abs(np.max(pbA)) if pbA.shape[0] > 0 else -1
            self.staticFeatures[14,i] = sign(np.max(pbA)) if pbA.shape[0] > 0 else 0
            self.staticFeatures[15,i] = abs(np.min(nbA)) if nbA.shape[0] > 0 else -1
            self.staticFeatures[16,i] = sign(np.min(nbA)) if nbA.shape[0] > 0 else 0
            self.staticFeatures[17,i] = abs(np.max(nbA)) if nbA.shape[0] > 0 else -1
            self.staticFeatures[18,i] = sign(np.max(nbA)) if nbA.shape[0] > 0 else 0
            
            # cost/constraint features
            cA = self.c[0,i]/self.A[abs(self.A[:,i]) > tol,i]
            self.staticFeatures[19,i] = abs(np.min(cA)) if self.c[0,i] >= 0 else -1
            self.staticFeatures[20,i] = sign(np.min(cA)) if self.c[0,i] >= 0 else 0
            self.staticFeatures[21,i] = abs(np.max(cA)) if self.c[0,i] >= 0 else -1
            self.staticFeatures[22,i] = sign(np.max(cA)) if self.c[0,i] >= 0 else 0
            self.staticFeatures[23,i] = abs(np.min(cA)) if self.c[0,i] < 0 else -1
            self.staticFeatures[24,i] = sign(np.min(cA)) if self.c[0,i] < 0 else 0
            self.staticFeatures[25,i] = abs(np.max(cA)) if self.c[0,i] < 0 else -1
            self.staticFeatures[26,i] = sign(np.max(cA)) if self.c[0,i] < 0 else 0
    
    def preprocessFeatures(self):
        sp = np.sum(self.state[0,1:])
        sa = np.sum(abs(self.state[0,1:]))
        self.c_pos = (sa + sp)/2
        self.c_neg = (sa - sp)/2
        
        self.A_pos = np.zeros([self.m,1])
        self.A_neg = np.zeros([self.m,1])
        for j in range(self.m):
            sp = np.sum(self.state[1+j,1:])
            sa = np.sum(abs(self.state[1+j,1:]))
            self.A_pos[j,0] = (sa + sp)/2
            self.A_neg[j,0] = (sa - sp)/2
    
    def features(self, act, mode=0, tol=1e-8):
        if mode == 0:
            dynamicFeatures = np.zeros([23])
        elif mode == 1:
            dynamicFeatures = np.zeros([28])
        else:
            raise NotImplementedError("Mode not recognized")
        
        # cost function features
        dynamicFeatures[0] = sign(self.state[0,1+act])
        dynamicFeatures[1] = abs(self.state[0,1+act])/self.c_pos
        dynamicFeatures[2] = abs(self.state[0,1+act])/self.c_neg
        
        # constraint coefficient features
        mask = self.state[:,1+act]>tol
        mask[0] = False
        ppA = self.state[mask,1+act]/self.A_pos
        pnA = self.state[mask,1+act]/self.A_neg
        mask = self.state[:,1+act]<-tol
        mask[0] = False
        npA = self.state[mask,1+act]/self.A_pos
        nnA = self.state[mask,1+act]/self.A_neg
        
        dynamicFeatures[3] = np.min(ppA) if ppA.shape[1] > 0 else -1
        dynamicFeatures[4] = np.max(ppA) if ppA.shape[1] > 0 else -1
        dynamicFeatures[5] = np.max(pnA) if pnA.shape[1] > 0 else -1
        dynamicFeatures[6] = np.max(pnA) if pnA.shape[1] > 0 else -1
        dynamicFeatures[7] = -np.min(npA) if npA.shape[1] > 0 else -1
        dynamicFeatures[8] = -np.max(npA) if npA.shape[1] > 0 else -1
        dynamicFeatures[9] = -np.max(nnA) if nnA.shape[1] > 0 else -1
        dynamicFeatures[10] = -np.max(nnA) if nnA.shape[1] > 0 else -1
        
        # bounds/constraint features
        mask = self.state[:,0]>tol
        mask[0] = False
        pbA = self.state[mask,1+act]/self.state[mask,0]
        dynamicFeatures[11] = abs(np.min(pbA)) if pbA.shape[0] > 0 else -1
        dynamicFeatures[12] = sign(np.min(pbA)) if pbA.shape[0] > 0 else 0
        dynamicFeatures[13] = abs(np.max(pbA)) if pbA.shape[0] > 0 else -1
        dynamicFeatures[14] = sign(np.max(pbA)) if pbA.shape[0] > 0 else 0
        
        # cost/constraint features
        mask = abs(self.state[:,1+act]) > tol
        mask[0] = False
        cA = self.state[0,1+act]/self.state[mask,1+act]
        dynamicFeatures[15] = abs(np.min(cA)) if self.state[0,1+act] >= 0 else -1
        dynamicFeatures[16] = sign(np.min(cA)) if self.state[0,1+act] >= 0 else 0
        dynamicFeatures[17] = abs(np.max(cA)) if self.state[0,1+act] >= 0 else -1
        dynamicFeatures[18] = sign(np.max(cA)) if self.state[0,1+act] >= 0 else 0
        dynamicFeatures[19] = abs(np.min(cA)) if self.state[0,1+act] < 0 else -1
        dynamicFeatures[20] = sign(np.min(cA)) if self.state[0,1+act] < 0 else 0
        dynamicFeatures[21] = abs(np.max(cA)) if self.state[0,1+act] < 0 else -1
        dynamicFeatures[22] = sign(np.max(cA)) if self.state[0,1+act] < 0 else 0
    
        if mode == 1:
            # bounds/constraint features : test ratio
            mask = (self.state[:,1+act]>tol) & (self.state[:,0]>tol)
            mask[0] = False
            bA = self.state[mask,1+act]/self.state[mask,0]
            mbA = np.min(bA)
            dynamicFeatures[23] = mbA
            dynamicFeatures[24] = np.max(bA)
            
            # additional cost measures : steepest edge, greatest improvement, least entered
            dynamicFeatures[25] = self.state[0,1+act]/np.linalg.norm(self.state[1:,1+act])
            dynamicFeatures[26] = self.entered[act]/self.steps
            dynamicFeatures[27] = self.state[0,1+act]*mbA
    
        return np.concatenate([self.staticFeatures[:,act],dynamicFeatures])
    
    def __init__(self, A, b, c, type = 'polyhedron'):
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
            - type (str) - type of the polyhedron : - 'polyhedron' is the general type (when no type is specified)
                                                    - 'unitcube' is a unit cube
                                                    - 'spindle' is a spindle
        """
        self.m = b.shape[0]
        self.n = c.shape[1]
        
        self.A = A
        self.b = b
        self.c = c
        
        self.initFeatures()
        
        self.type = type

    def getNumberOfActions(self): return self.n

    def sizeOfActionVector(self, mode=0):
        return self.n if mode == 0 else 1

    def cube(n,obj='random'):
        """
        Instantiation of a Simplex-Polyhedra on a
        n-dimensional cube
        Parameters :
            - n (integer) - dimension of the cube
            - obj (str) - 'random' for a random objective
                        - 'pos_random' for a random positive objective
                        - 'neg_random' for a random negative objective
                        - 'unit' for an unit objective (sum of coordinates)
                        - 'negunit' for the opposite unit objective (-sum of coordinates)
        """
        A,b = poly.cube(n)
        c = poly.objective(n,obj)
            
        A,b,c = poly.standardForm(A,b,c)
        
        return SimPolyhedra(A,b,c,'unitcube')
        
    def randomSpindle(n,cond=100,obj='random'):
        """
        Instantiation of a Simplex-Polyhedra on a
        n-dimensional spindle
        Parameters :
            - n (integer) - dimension of the spindle
            - cond (float) - condition number of the spindle
            - obj (str) - 'random' for a random objective
                        - 'pos_random' for a random positive objective
                        - 'neg_random' for a random negative objective
                        - 'unit' for an unit objective (sum of coordinates)
                        - 'neg_unit' for the opposite unit objective (-sum of coordinates)
        """
        A,b = poly.spindle_random(n,cond)
        c = poly.objective(n,obj)
            
        A,b,c = poly.standardForm(A,b,c)
        
        return SimPolyhedra(A,b,c,'spindle')
        
    def randomPolyhedron(n):
        A = np.random.random([2*n,n])
        b = np.zeros([2*n,1])
        b[:,0] = np.sum(A,axis=1)/2.
        c = poly.objective(n,'neg_random')
            
        A,b,c = poly.standardForm(A,b,c)
        
        return SimPolyhedra(A,b,c)
        
    def reset(self,initBasis=None,randomSteps=0):
        """
        Sample and return an initial state
        (may be fixed)
        """
        if initBasis is None:
            self.basis = self.feasibleBasis()
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
        
        self.entered = [0 for k in range(self.n)]
        self.steps = 0
        
        # Creation of the state (containing the entire simplex "tableau" )
        self.state = np.vstack([np.hstack([z_r,c_r]),np.hstack([b_r,A_r])])

        self.preprocessFeatures()
        
        # Random steps to find a random basis
        if initBasis is None and randomSteps == 0:
            randomSteps = self.autoRandomSteps()
            
        for k in range(randomSteps):
            a = np.random.choice(self.getAvailableActions())
            self.step(a,False)
        
        return self.observe()

    def feasibleBasis(self):
        if self.type == 'polyhedron':
            return feasibleBasis(self.A,self.b)
        elif self.type == 'unitcube':
            return poly.cube_randomBasis(self.n-self.m)
        elif self.type == 'spindle':
            return poly.spindle_randomExtremityBasis(self.n-self.m)
        
    def autoRandomSteps(self):
        if self.type == 'polyhedron':
            return 0
        elif self.type == 'unitcube':
            return 0
        elif self.type == 'spindle':
            return poly.spindle_randomStepsCount(self.n-self.m)
        
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

    def postprocess_action(self,a,mode=0):
        if mode == 0:
            return self.actionToOneHot(a)
        elif mode == 1:
            return [self.actionToDistanceToRCost(a)]

        raise NotImplementedError("Mode not recognized")

    def actionToOneHot(self,a):
        return mpu.ml.indices2one_hot([a], nb_classes=self.n)[0]

    def actionToDistanceToRCost(self,a):
        return np.square(np.min(self.state[0,1:]) - self.state[0,a+1])
        
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
            if not self.basis[act]:
                c = self.state[0,1+act]/np.linalg.norm(self.state[1:,1+act])
                if c < min_c:
                    min_c = c
                    argmin_c = act
        
        return argmin_c
        
    def testAction(self):
        min_c = np.inf
        argmin_c = None
        for act in range(self.n):
            if not self.basis[act]:
                c = self.state[0,1+act]/np.max(self.state[1:,1+act])
                if c < min_c:
                    min_c = c
                    argmin_c = act
        
        return argmin_c
        
    def leastEnteredAction(self):
        min_e = np.inf
        argmin_e = None
        tie = None
        for act in range(self.n):
            if not self.basis[act] and self.state[0,1+act] <= 0:
                e = self.entered[act]
                if e < min_e:
                    min_e = e
                    argmin_e = act
                    tie = self.state[0,1+act]/np.linalg.norm(self.state[1:,1+act])
                elif e == min_e:
                    c = self.state[0,1+act]/np.linalg.norm(self.state[1:,1+act])
                    if c < tie:
                        min_e = e
                        argmin_e = act
                        tie = c
        return argmin_e
        
    def postprocess_action(self,a,mode=0):
        if mode == 0:
            return self.actionToOneHot(a)
        elif mode == 1:
            return self.actionToDistanceToRCost(a)

        raise NotImplementedError("Mode not recognized")

    def actionToOneHot(self,a):
        return mpu.ml.indices2one_hot([a], nb_classes=self.n)[0]

    def actionToDistanceToRCost(self,a):
        return np.square(np.min(self.state[0,1:]) - self.state[0,a+1])
        
    def step(self, act, trueStep=True):
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
            for i in range(self.m+1):
                if i != 1+e:
                    self.state[i,:] -= self.state[1+e,:]*self.state[i,1+act]/self.state[1+e,1+act]
            self.state[1+e,:] /= self.state[1+e,1+act]
            
            self.preprocessFeatures()
            
            # Update of informations on basic variables
            self.basis[self.rowVar[e]] = False
            self.basis[act] = True
            self.rowVar[e] = act
            
            if trueStep:
                self.entered[act] += 1
                self.steps += 1
            
            # Termination if optimal
            if self.isOptimal():
                reward = 1
                done = True
                
        return self.observe(), reward, done, {}
 
    def resetHistory(self):
        self.entered = [0 for k in range(self.n)]
        
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
            
    def isOptimal(self,tol=1e-8):
        return (self.state[0,1:]+tol >= 0).all()
    
    
if __name__ == '__main__':
    n = 50
    P = SimPolyhedra.cube(n)
    """ 
    2 ways to initialize a basis
    automatically (with no input) :
    """
    P.reset()
    """
    or directly by specifying a basis :
    """
    #P.reset(poly.cube_randomBasis(n))
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

    """ Test rule """
    P.reset(reference_basis)
    steps = 0
    acts = []
    while not P.isOptimal():
        print("Objective = " + str(P.objective()))
        a = P.testAction()
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
    
    """ Least entered rule """
    P.reset(reference_basis)
    steps = 0
    acts = []
    while not P.isOptimal():
        print("Objective = " + str(P.objective()))
        a = P.leastEnteredAction()
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

    """ Random rule """
    P.reset(reference_basis)
    steps = 0
    acts = []
    while not P.isOptimal():
        print("Objective = " + str(P.objective()))
        a = P.randomImprovingAction()
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
