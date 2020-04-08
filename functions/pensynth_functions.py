### penalized synthetic control main functions
### 8/04/2020
### Jérémy L'Hour

import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import minimize


### FUNCTION: pensynth_weights 
### compute penalized synthetic control weights

def pensynth_weights(X0, X1, pen = .1, **kwargs):
    V = kwargs.get('V', np.identity(X0.shape[0]))
    # OBJECTIVE
    n = X0.shape[1]
    delta = np.diag((X0 - np.reshape(np.repeat(X1,n,axis=0), (X1.shape[0],n))).T.dot(V.dot(X0 - np.reshape(np.repeat(X1,n,axis=0), (X1.shape[0],n)))))
    P = (X0.T).dot(V.dot(X0))
    P = matrix(P)
    q = -X0.T.dot(V.dot(X1)) + (pen/2)*delta
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

### FUNCTION: pensynth 
### Main penalized synthetic control function
### compute the solution for each treated, for a given lambda

def pensynth(X0, X1, Y0, Y1, pen = .1, **kwargs):
    V = kwargs.get('V', np.identity(X0.shape[0]))
    X0 = np.array(X0)
    X1 = np.array(X1)
    n1 = X1.shape[1]
    weights = []
    for i in range(n1):
        sol = pensynth_weights(X0, X1[:,i], pen = pen, V = V)
        weights.append(np.array(sol))
    weights = np.squeeze(weights)
    indiv_TE = Y1 - weights.dot(Y0)
    return np.mean(indiv_TE), indiv_TE, weights


X1 = np.array([[0,1],[0,1]])
X0 = np.array([[0,1,-1],[1,0,-1]])
Y1 = np.array([5,6])
Y0 = np.array([3,6,7])
pensynth(X0, X1, Y0, Y1, pen = .1)