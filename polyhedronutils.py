import math
import numpy as np

def standardForm(A,b,c):
    """
        Passes a linear program of the form :
        min c'x
        Ax <= b
        x >= 0
        to the standard form :
        min c'x + 0'y
        Ax + Iy = b
        x,y >= 0
    """
    m = A.shape[0]
    A = np.hstack([A,np.eye(m)])
    c = np.hstack([c,np.zeros([1,m])])
    return A,b,c

def objective(n,obj):
    if obj == 'random':
        return np.random.random([1,n])-0.5
    elif obj == 'pos_random':
        return np.random.random([1,n])
    elif obj == 'neg_random':
        return -np.random.random([1,n])
    elif obj == 'unit':
        return np.ones([1,n])
    elif obj == 'neg_unit':
        return -np.ones([1,n])
    elif obj == 'zero':
        return np.zeros([1,n])
    
def conditionNumber(A):
    U,D,Vh = np.linalg.svd(A)
    return D[0]/D[-1]
    
def cube(n):
    """
        Generates the n-dimensional unit cube
        Parameters : 
            - n (integer) : dimension of the cube
    """
    A = np.eye(n)
    b = np.ones([n,1])
    return A,b
  
def cube_randomBasis(n):
    """
        Generates a random feasible basis
        for the n-dimensional unit cube
        Parameters : 
            - n (integer) : dimension of the cube
    """
    half_basis = np.random.choice(a=[False, True], size=n)
    return np.hstack([half_basis, ~half_basis]).tolist()
    
def cube_originBasis(n):
    """
        Generates a random feasible basis
        for the n-dimensional unit cube
        Parameters : 
            - n (integer) : dimension of the cube
    """
    return [False]*n + [True]*n
    
def cube_unitBasis(n):
    """
        Generates a random feasible basis
        for the n-dimensional unit cube
        Parameters : 
            - n (integer) : dimension of the cube
    """
    return [True]*n + [False]*n
    
def spindle_random(n,K=10):
    """
        Generates a random n-dimensional spindle
        Parameters : 
            - n (integer) : dimension of the spindle
            - K=10 (float) : condition number of the problem
    """
    A = np.zeros([n,n])
    b = np.ones([n,1])

    # First point of spindle is fixed at x = (0,...,0)
    # First cone radiuses are the unit vectors
    O1 = np.zeros([n,1])
    
    # Second point of spindle is selected randomly in the first cone
    # (i.e all coordinates > 0)
    O2 = np.random.random([n,1])
    
    # Second cone radiuses are generated randomly and are negative 
    # to ensure that the resulting polyhedron will be bounded
    S = -np.random.random([n,n])
    
    # We want to control the condition number so we mess a bit
    # with singular values of the matrix of radiuses
    #U,D,Vh = np.linalg.svd(S)
    #K_S = D[0]/D[n-1]
    #k = math.log(K)/math.log(K_S)
    #D **= k
    #if K > 1:
    #    eps = (D[0] - K*D[n-1])/(K-1)
    #    D = D + eps
    #else:
    #    D = np.ones(D.shape)
    #S = U*D*Vh
    
    # Translation of y such that x = (0,...,0) is in the cone of origin y and radiuses S[:,i]
    sigma = np.random.random([n,1])
    dO = S.dot(sigma) + O2
    O2 += O1-dO
    O2 = np.reshape(O2,[n])
    
    # First set of n inequalities is given by unit vectors i.e x_i >= 0 for all i
    # Second set of n inequalities is given by the radiuses S[:,i]
    # More precisely, to get the inequality opposed to S[:,i], 
    # we find alpha such that alpha'x = 1 where x = O2 + sigma_1*S[:,1] + ... + sigma_(i-1)*S[:,i-1] + sigma_(i+1)*S[:,i+1] + ... + sigma_n*S[:,n]
    # (to get n vectors x, we input sigma as the unit vectors)
    for i in range(n):
        M = np.zeros([n,n])
        for k in range(n):
            if k != i:
                M[k,:] = (O2 + S[:,k].transpose())
        M[i,:] = O2
        alpha = np.linalg.inv(M).dot(np.ones([n,1]))
        
        alpha = np.reshape(alpha,[n])
        
        # Here, we know O1 = 0, and O1 is in the second cone so alpha'O1 is supposed to verify the inequality
        # We also know that alpha' * O1 = 0 < 1, thus we know the inequality is alpha'x <= 1
        A[i,:] = alpha
        
    return A,b
    
def spindle_randomExtremityBasis(n):
    """
        Gives the basis of one of the two extreme vertices
        of a n-dimensional spindle, chosen randomly
        Parameters : 
            - n (integer) : dimension of the spindle
    """
    flag = np.random.random() < 0.5
    return [flag]*n + [not flag]*n
    
def spindle_randomStepsCount(n):
    """
        Gives a good amount of steps to get a random
        enough basis from an initial basis in a 
        n-dimensional spindle
        Parameters : 
            - n (integer) : dimension of the spindle
    """
    return 4*n
    