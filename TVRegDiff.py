#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:48:28 2018

@author: anastasia
"""

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy import array, linalg, dot
import scipy as sp
import matplotlib.pyplot as plt
import scipy.integrate as integrate

    
    
def diff(vector):
    return list([ x-y for (x,y) in zip(vector[1:],vector[:-1]) ])
    

def TVRegDiff( data, iter1, alph, u0 = 0, scale = 'large', ep = 1e-6, dx = 0, plotflag = 1, diagflag = 1):
#data        Vector of data to be differentiated.
#iter1       Number of iterations to run the main loop.  A stopping
#            condition based on the norm of the gradient vector g
#            below would be an easy modification.  No default value.
#alph        Regularization parameter.  This is the main parameter
                   #to fiddle with.  Start by varying by orders of
                   #magnitude until reasonable results are obtained.  A
                   #value to the nearest power of 10 is usally adequate.
                   #No default value.  Higher values increase
                   #regularization strenght and improve conditioning.
    
    
    n = len(data)
    dx = 1/n
    

    def create_A(n):
        A = np.zeros((n,n))
        num = 1
        for i in range(0,n):
            for j in range(0, num):
                A[i][j] = 1
            num = num + 1

        return A

    def create_AT(n):
        AT = np.ones((n,n))
        j = 0
        for i in range(1,n):
            AT[i][j] = 0
            j = j + 1
        return AT
        
    A = create_A(n)
    AT = create_AT(n)    
        
    
    #Construct differentiation matrix.
    c = np.ones(n)
    D = spdiags( [ -c, c ], [ 0, 1 ], n, n ) / dx
    D = D.todense()
    D[n-1, n-1] = 0 
    
    DT = D.T
    data = data-data[0]
    
    diff_vec = diff(data)
    
    u0 = np.zeros(len( diff_vec )+1)
    for x in range(1, len( diff_vec )+1):
        u0[x] =  diff_vec[x-1]

    u = u0
    ATd = AT @ data 

        #Main loop.
        
    gradient_matrix = []
    cost_matrix = []
    
    for ii in range(1, iter1+1):
        #Diagonal matrix of weights, for linearizing E-L equation.
        
        Qelement1 = np.sqrt( np.square( D @ u ) +  ep )
        
        Qelement2 = 1/Qelement1
        
        Q = spdiags(Qelement2, 0, n, n )
        
        Q = Q.todense()
        
        L = DT @ Q @ D
        
        g = AT @ (A @ u) - ATd 
        
        g= g + alph * L @ u

        n_vector = np.ones(n)
        
        for x in range(1, n+1):
            n_vector[x-1] = int(n+1 -x)
        
        c = np.cumsum(n_vector)
        c = c.T

        B = alph * L + spdiags(list(reversed(c)) , 0, n, n )    
            
        #R = linalg.cholesky(B)

        tol = 1.0e-4;
        maxit = 100;
        
            
        
        #print(ii)
        
        
        if diagflag:
                
           s = sp.sparse.linalg.cg((alph * L + AT @ A), b = (-g).T, M = B,  tol = tol, maxiter = maxit) 
           gradient_matrix.append(np.linalg.norm(g))
            
        u = u + s[0];
        
        
        #Ru = sum(np.square(u))
        #Ru = sum(np.square(diff(u)))
        
        #Au = np.cumsum(u)
         
        #DF = sum(np.square(Au-data))
        
        #cost = alph * Ru + DF
        
        #cost_matrix.append(cost)

    return u


