import numpy as np

def standardForm(A,b,c):
    m = A.shape[0]
    A = np.hstack([A,np.eye(m)])
    c = np.hstack([c,np.zeros([1,m])])
    return A,b,c

def cube(n):
	A = np.eye(n)
	b = np.ones([n,1])
	return A,b