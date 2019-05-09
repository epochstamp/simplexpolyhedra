import sys
import math
import gym
import numpy as np
from scipy import optimize
import scipy
import time

def feasibleBasis(A,b):
	m = A.shape[0]
	n = A.shape[1]
	A_p = np.hstack([A,np.eye(m)])
	for i in range(m):
		if b[i] < 0:
			A_p[i,n+i] = -1.
	
	c = np.hstack([np.zeros([1,n]),np.ones([1,m])])
	
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
		return basis
	else:
		print("Error : no feasible basis found")
	
	

class SimPolyhedra():

	"""
	Discount factor
	"""
	def gamma(self): return 0.95

	def __init__(self, A, b, c):
		"""
		Instantation of a Simplex-Polyhedra problem
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

	def reset(self):
		"""
		Sample and return an initial state
		(may be fixed)
		"""
		self.basis = feasibleBasis(self.A,self.b)
		A_B_inv = np.linalg.inv(self.A[:,self.basis])

		A_r = A_B_inv.dot(self.A)
		b_r = A_B_inv.dot(self.b)
		c_r = self.c - self.c[0,self.basis].dot(A_r)
		z_r = -self.c[:,self.basis].dot(b_r)
		
		self.rowVar = []
		for i in range(self.n):
			if self.basis[i]:
				self.rowVar += [i]
		
		self.state = np.vstack([np.hstack([z_r,c_r]),np.hstack([b_r,A_r])])
		
		return self.observe()
    
	def step(self, act):
		"""
		Transition step from state to successor given a discrete action
		Parameters : 
			- act \in [0...dim-1] : index of a variable (tableau column)
		"""
		reward = 0
		done = False
		if not self.basis[act]:
			e,m = -1,np.inf
			for i in range(self.m):
				if self.state[1+i,1+act] > 0:
					r = self.state[1+i,0]/self.state[1+i,1+act]
					if r <= m:
						m = r
						e = i
					
			assert(e != -1)
			
			E = np.eye(self.m+1)
			E[:,1+e] = -self.state[:,1+act]/self.state[1+e,1+act]
			E[1+e,1+e] = 1./self.state[1+e,1+act]
			
			self.state = E.dot(self.state)
			
			self.basis[self.rowVar[e]] = False
			self.basis[act] = True
			self.rowVar[e] = act
			
			if (self.state[0,1:] >= 0).all():
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
			
