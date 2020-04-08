### penalized synthetic control main functions
### 8/04/2020
### Jérémy L'Hour

import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import minimize

# X1 = np.array([[1,.5],[2,2.5]])
X1 = np.array([1,2])
X0 = np.array([[0,3,1.6],[0,4,2.5]])


### compute penalized synthetic control weights

def pensynth_weights(X0, X1, V, pen = .1):
    # OBJECTIVE
    n = X0.shape[1]
    delta = np.diag((X0 - np.repeat(X1,n,axis=1)).T.dot(V.dot(X0 - np.repeat(X1,n,axis=1))))
    delta = np.reshape(delta,(n,1))
    P = (X0.T).dot(V.dot(X0))
    P = matrix(P)
    q = -X0.T.dot(V.dot(X1)) + (pen/2)*delta
    q = q.astype('float')
    q = matrix(q)
    # ADDING-UP
    A = matrix(1.0,(1,n))
    b = matrix(1.0)
    # NON-NEGATIVITY
    G = matrix(np.concatenate((np.identity(n),-np.identity(n))))
    h = matrix(np.concatenate((np.ones(n),np.zeros(n))))
    # SOLUTION
    sol = solvers.qp(P,q,G,h,A,b)
    return sol['x']   

### Main penalized synthetic control function
### compute the solution for a given lambda

def pensynth(X0, X1, Y0, Y0, V, pen = .1):
    X0 = np.array(X0)
    X1 = np.array(X1)
    n0 = X0.shape[1]
    n1 = X1.shape[1]
    weights = []
    for i in range(n1):
        sol = pensynth_weights(X0, X1[:,0], V=np.identity(2), pen = .1)
        weights.append(np.array(sol))

