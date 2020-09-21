#!/usr/bin/env python
# coding: utf-8

# # Simple iteration for systems of linear equations

# First, generate a random diagonally dominant matrix, for testing.

# In[4]:


import numpy as np
rndm = np.random.RandomState(1234)

n = 10
A = rndm.uniform(size=(n, n)) + np.diagflat([15]*n)
b = rndm.uniform(size=n)
A


# # I.  Jacobi iteration
# 
# Given
# 
# $$
# A x = b
# $$
# 
# separate the diagonal part $D$,
# 
# $$ A = D + (A - D) $$
# 
# and write
# 
# $$
# x = D^{-1} (D - A) x + D^{-1} b\;.
# $$
# 
# Then iterate
# 
# $$
# x_{n + 1} = B x_{n} + c\;,
# $$
# 
# where 
# 
# $$
# B = D^{-1} (A - D) \qquad \text{and} \qquad c = D^{-1} b
# $$
# 

# Let's construct the matrix and the r.h.s. for the Jacobi iteration

# In[6]:


diag_1d = np.diag(A)

B = -A.copy()
np.fill_diagonal(B, 0)

D = np.diag(diag_1d)
invD = np.diag(1./diag_1d)
BB = invD @ B 
c = invD @ b
B


# In[7]:


# sanity checks
from numpy.testing import assert_allclose

assert_allclose(-B + D, A)


# xx is a "ground truth" solution, compute it using a direct method
xx = np.linalg.solve(A, b)

np.testing.assert_allclose(A@xx, b)
np.testing.assert_allclose(D@xx, B@xx + b)
np.testing.assert_allclose(xx, BB@xx + c)


# Check that $\| B\| \leqslant 1$:

# In[8]:


np.linalg.norm(BB)


# ### Do the Jacobi iteration

# In[9]:


n_iter = 50

x0 = np.ones(n)
x = x0
for _ in range(n_iter):
    x = BB @ x + c


# In[10]:


# Check the result:

A @ x - b


# ### Task I.1
# 
# Collect the proof-of-concept above into a single function implementing the Jacobi iteration. This function should receive the r.h.s. matrix $A$, the l.h.s. vector `b`, and the number of iterations to perform.
# 
# 
# The matrix $A$ in the illustration above is strongly diagonally dominant, by construction. 
# What happens if the diagonal matrix elements of $A$ are made smaller? Check the convergence of the Jacobi iteration, and check the value of the norm of $B$.
# 
# (20% of the total grade)
# 

# In[ ]:


# ... ENTER YOUR CODE HERE ...
def jacobi_iteration(x,A,b,iter_num=1000):
    #matrix D
    diag_1d = np.diag(A)
    D = np.diag(diag_1d)
    #inverse of D
    invD = np.diag(1./diag_1d)
    #matrix B
    B = -A.copy()
    np.fill_diagonal(B, 0)
    BB = invD @ B 
    #rhs vector
    c = invD @ b
    #iterate
    iters=1
    while iters<iter_num:
        x=np.dot(BB,x)+c
        iters=iters+1
    return x

x=jacobi_iteration(x0,A,b,iter_num=1000)   
x


# # II. Seidel's iteration.

# ##### Task II.1
# 
# Implement the Seidel's iteration. 
# 
# Test it on a random matrix. Study the convergence of iterations, relate to the norm of the iteration matrix.
# 
# (30% of the total grade)

# In[ ]:


# ... ENTER YOUR CODE HERE ...


# # III. Minimum residual scheme

# ### Task III.1
# 
# Implement the $\textit{minimum residual}$ scheme: an explicit non-stationary method, where at each step you select the iteration parameter $\tau_n$ to minimize the residual $\mathbf{r}_{n+1}$ given $\mathbf{r}_n$. Test it on a random matrix, study the convergence to the solution, in terms of the norm of the residual and the deviation from the ground truth solution (which you can obtain using a direct method). Study how the iteration parameter $\tau_n$ changes as iterations progress.
# 
# (50% of the grade)

# In[ ]:


# ... ENTER YOUR CODE HERE ...
def mrm_iteration(x,A,b,iter_num=1000):
    iters=0
    x,b=x.reshape([-1,1]),b.reshape([-1,1])
    while iters<iter_num:
        #current residual
        r=(np.dot(A,x)-b).reshape([-1,1])
        #Ar
        Ar=np.dot(A,r)
        #current tau
        tau=np.sum(np.dot(r.T,Ar))/np.sum(Ar**2)
        #update
        x=x-tau*r
        iters=iters+1
    return x
mrm_iteration(x0,A,b,iter_num=1000)