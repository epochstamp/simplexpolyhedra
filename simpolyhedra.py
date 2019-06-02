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
    def gamma(self): return 0.995

    def initFeatures(self,tol = 1e-8):
        self.staticFeatures = np.zeros([27,self.n])
        #First one includes static feature size, the others are dynamic feature size
        self.featureSizes = [27,23,5,1]
        
        c_pos = np.sum(self.c[self.c > 0])
        c_neg = -np.sum(self.c[self.c < 0])
        
        A_pos = np.zeros([self.m,1])
        A_neg = np.zeros([self.m,1])
        for j in range(self.m):
            A_pos[j,0] = np.sum(self.A[j,self.A[j,:]>0])
            A_neg[j,0] = -np.sum(self.A[j,self.A[j,:]<0])
        
        for i in range(self.n):
            # cost function features
            self.staticFeatures[0,i] = sign(self.c[0,i])
            self.staticFeatures[1,i] = abs(self.c[0,i])/c_pos if c_pos > 0 else -1
            self.staticFeatures[2,i] = abs(self.c[0,i])/c_neg if c_neg > 0 else -1
            

            # constraint coefficient features
            mppA = np.inf
            MppA = -np.inf
            mpnA = np.inf
            MpnA = -np.inf
            mnpA = np.inf
            MnpA = -np.inf
            mnnA = np.inf
            MnnA = -np.inf
            for j in range(self.m):
                if self.A[j,i] > 0 :
                    if A_pos[j,0] > 0:
                        ppA = self.A[j,i]/A_pos[j,0]
                        if ppA > MppA:
                            MppA = ppA
                        if ppA < mppA:
                            mppA = ppA
                    elif A_neg[j,0] < 0:
                        pnA = self.A[j,i]/A_neg[j,0]
                        if pnA > MpnA:
                            MpnA = pnA
                        if pnA < mpnA:
                            mpnA = pnA
                elif self.A[j,i] < 0:
                    if A_pos[j,0] > 0:
                        npA = -self.A[j,i]/A_pos[j,0]
                        if npA > MnpA:
                            MnpA = npA
                        if npA < mnpA:
                            mnpA = npA
                    elif A_neg[j,0] < 0:
                        nnA = -self.A[j,i]/A_neg[j,0]
                        if nnA > MnnA:
                            MnnA = nnA
                        if nnA < mnnA:
                            mnnA = nnA
                
            self.staticFeatures[3,i] = mppA if mppA != np.inf else -1
            self.staticFeatures[4,i] = MppA if MppA != -np.inf else -1
            self.staticFeatures[5,i] = mpnA if mpnA != np.inf else -1
            self.staticFeatures[6,i] = MpnA if MpnA != -np.inf else -1
            self.staticFeatures[7,i] = mnpA if mnpA != np.inf else -1
            self.staticFeatures[8,i] = MnpA if MnpA != -np.inf else -1
            self.staticFeatures[9,i] = mnnA if mnnA != np.inf else -1
            self.staticFeatures[10,i] = MnnA if MnnA != -np.inf else -1
            
            # bounds/constraint features
            mpbA = np.inf
            MpbA = -np.inf
            mnbA = np.inf
            MnbA = -np.inf
            for j in range(self.m):
                if self.b[j,0] > 0:
                    pbA = self.A[j,i]/self.b[j,0]
                    if pbA > MpbA:
                        MpbA = pbA
                    if pbA < mpbA:
                        mpbA = pbA
                elif self.b[j,0] < 0:
                    nbA = self.A[j,i]/(-self.b[j,0])
                    if nbA > MnbA:
                        MnbA = nbA
                    if nbA < mnbA:
                        mnbA = nbA
                        
            self.staticFeatures[11,i] = abs(mpbA) if mpbA != np.inf else -1
            self.staticFeatures[12,i] = sign(mpbA) if mpbA != np.inf else 0
            self.staticFeatures[13,i] = abs(MpbA) if MpbA != -np.inf else -1
            self.staticFeatures[14,i] = sign(MpbA) if MpbA != -np.inf else 0
            self.staticFeatures[15,i] = abs(mnbA) if mnbA != np.inf else -1
            self.staticFeatures[16,i] = sign(mnbA) if mnbA != np.inf else 0
            self.staticFeatures[17,i] = abs(MnbA) if MnbA != -np.inf else -1
            self.staticFeatures[18,i] = sign(MnbA) if MnbA != -np.inf else 0
            
            # cost/constraint features
            assert((abs(self.A[:,i]) > tol).any())
            cA = abs(self.c[0,i])/self.A[abs(self.A[:,i]) > tol,i]
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

    """
    /!\ mode is a mask
    to enable/disable groups of features.
    e.g., 15 = 1111 enable all features,
    7 = 0111 enable all dynamics features and discard static features...
    """

    def getFeatureSize(self, mode=0):
        true_sizes = np.unpackbits(np.asarray([mode], dtype=np.uint8))[-len(self.featureSizes):]
        if true_sizes.shape[0] != len(self.featureSizes):
            raise NotImplementedError("Mode not recognized")
        return np.sum(true_sizes * self.featureSizes)

    
    def features(self, act, mode=0, tol=1e-8):
        true_sizes = np.unpackbits(np.asarray([mode], dtype=np.uint8))[-len(self.featureSizes):]
        if true_sizes.shape[0] != len(self.featureSizes):
            raise NotImplementedError("Mode not recognized")

        dynamicFeatures = np.zeros([np.sum(true_sizes[1:]*self.featureSizes[1:])])
        k = 0
        i = -1
        reduced_cost = self.state[0,1+act]
        if true_sizes[1] > 0:
            if i < 0 : i = 0 
            
            abs_reduced_cost = abs(reduced_cost)
            # cost function features
            dynamicFeatures[i+0] = sign(reduced_cost)
            dynamicFeatures[i+1] = abs_reduced_cost/self.c_pos if self.c_pos > 0 else -1
            dynamicFeatures[i+2] = abs_reduced_cost/self.c_neg if self.c_neg > 0 else -1
            
            # constraint coefficient features
            mppA = np.inf
            MppA = -np.inf
            mpnA = np.inf
            MpnA = -np.inf
            mnpA = np.inf
            MnpA = -np.inf
            mnnA = np.inf
            MnnA = -np.inf
            for j in range(self.m):
                v_per_line = self.state[1+j,1+act]
                if v_per_line > tol :
                    if self.A_pos[j,0] > tol:
                        ppA = v_per_line/self.A_pos[j,0]
                        if ppA > MppA:
                            MppA = ppA
                        if ppA < mppA:
                            mppA = ppA
                    elif self.A_neg[j,0] < -tol:
                        pnA = v_per_line/self.A_neg[j,0]
                        if pnA > MpnA:
                            MpnA = pnA
                        if pnA < mpnA:
                            mpnA = pnA
                elif v_per_line < -tol:
                    if self.A_pos[j,0] > tol:
                        npA = -v_per_line/self.A_pos[j,0]
                        if npA > MnpA:
                            MnpA = npA
                        if npA < mnpA:
                            mnpA = npA
                    elif self.A_neg[j,0] < -tol:
                        nnA = -v_per_line/self.A_neg[j,0]
                        if nnA > MnnA:
                            MnnA = nnA
                        if nnA < mnnA:
                            mnnA = nnA
            
            dynamicFeatures[i+3] = mppA if mppA != np.inf else -1
            dynamicFeatures[i+4] = MppA if MppA != -np.inf else -1
            dynamicFeatures[i+5] = mpnA if mpnA != np.inf else -1
            dynamicFeatures[i+6] = MpnA if MpnA != -np.inf else -1
            dynamicFeatures[i+7] = mnpA if mnpA != np.inf else -1
            dynamicFeatures[i+8] = MnpA if MnpA != -np.inf else -1
            dynamicFeatures[i+9] = mnnA if mnnA != np.inf else -1
            dynamicFeatures[i+10] = MnnA if MnnA != -np.inf else -1
            
            # bounds/constraint features
            mpbA = np.inf
            MpbA = -np.inf
            mnbA = np.inf
            MnbA = -np.inf
            for j in range(self.m):
                v_per_line = self.state[1+j,1+act]
                if self.state[j,0] > tol:
                    pbA = v_per_line/self.state[j,0]
                    if pbA > MpbA:
                        MpbA = pbA
                    if pbA < mpbA:
                        mpbA = pbA
                elif self.state[j,0] < -tol:
                    nbA = v_per_line/(-self.state[j,0])
                    if nbA > MnbA:
                        MnbA = nbA
                    if nbA < mnbA:
                        mnbA = nbA
            abs_mpbA = abs(mpbA)
            sign_mpbA = sign(mpbA)
            abs_MpbA = abs(MpbA)
            sign_MpbA = sign(MpbA) 

            
            
            
                     
      
            dynamicFeatures[i+11] = abs_mpbA if mpbA != np.inf else -1
            dynamicFeatures[i+12] = sign_mpbA if mpbA != np.inf else 0
            dynamicFeatures[i+13] = abs_mpbA if MpbA != -np.inf else -1
            dynamicFeatures[i+14] = sign_mpbA if MpbA != -np.inf else 0
                
            # cost/constraint features
            # assert((abs(self.state[:,1+act]) > tol).any())
            mask = abs(self.state[:,1+act]) > tol
            mask[0] = False
            cA = abs(reduced_cost)/self.state[mask,1+act]
            min_cA = np.min(cA)
            max_cA = np.max(cA)
            signmin_cA = sign(min_cA)
            absmin_cA = np.abs(min_cA)
            signmax_cA = sign(max_cA)
            absmax_cA = np.abs(max_cA)
            poscond = reduced_cost >= 0
            negcond = reduced_cost < 0
            dynamicFeatures[i+15] = absmin_cA if poscond else -1
            dynamicFeatures[i+16] = signmin_cA if poscond else 0
            dynamicFeatures[i+17] = absmax_cA if poscond else -1
            dynamicFeatures[i+18] = signmax_cA if poscond else 0
            dynamicFeatures[i+19] = absmin_cA if negcond else -1
            dynamicFeatures[i+20] = signmin_cA if negcond else 0
            dynamicFeatures[i+21] = absmax_cA if negcond else -1
            dynamicFeatures[i+22] = signmax_cA if negcond else 0
            i += self.featureSizes[1]
    
        if true_sizes[2] > 0:
            if i > 0: i = 0
            # bounds/constraint features : test ratio
            mask = (self.state[:,1+act]>tol) & (self.state[:,0]>tol)
            mask[0] = False
            bA = self.state[mask,1+act]/self.state[mask,0]
            mbA = np.min(bA)
            dynamicFeatures[i+0] = mbA
            dynamicFeatures[i+1] = np.max(bA)
            
            # additional cost measures : steepest edge, greatest improvement, least entered
            dynamicFeatures[i+2] = reduced_cost/np.linalg.norm(self.state[1:,1+act])
            dynamicFeatures[i+3] = self.entered[act]/self.steps
            dynamicFeatures[i+4] = reduced_cost*mbA
            i += self.featureSizes[2]
        if true_sizes[3] > 0:
            if i < 0: i = 0
            dynamicFeatures[i+0] = (np.min(self.state[0,1:])/reduced_cost)**2 if abs(reduced_cost) > 0 else -1 #np.square(reduced_cost-np.min(self.state[0,1:]))
        return np.concatenate([self.staticFeatures[:,act],dynamicFeatures]) if true_sizes[0] > 0 else dynamicFeatures
    
    def __init__(self, A, b, c, type = 'polyhedron', type_rew="montecarlo"):
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
        self.type_rew = type_rew
        self.type = type

    def getNumberOfActions(self): return self.n

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
        self.steps = 1
        
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

    def getType(self) : return self.type

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

    def reflexAction(self):
        if self.type == "unitcube":
            return self.dantzigAction()
        elif self.type == "spindle":
            return self.steepestEdgeAction()
        return self.dantzigAction()


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

            done = self.isOptimal()
            if self.type_rew == "montecarlo":
                # Terminal reward when optimal else 0
                reward = 0 if not done else 1#0.01 * -self.entered[act]/self.steps - 0.01 * sign(self.state[0,1+act])
            elif self.type_rew == "negtick":
                reward = -1 if not done else 1
            elif self.type_rew == "small_penalties":
                reward = -0.01 * (self.state[0,1+act]/np.linalg.norm(self.state[0,:]))
            else: 
                raise NotImplementedError("Reward mode not recognized.")
                
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
    n = 150
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
