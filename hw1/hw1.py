#!/usr/bin/env python
# coding: utf-8

# # I. $LU$ factorization of a square matrix

# Consider a simple naive implementation of the LU decomposition. 
# 
# Note that we're using the `numpy` arrays to represent matrices [do **not** use `np.matrix`].

# In[1]:


import numpy as np

def diy_lu(a):
    """Construct the LU decomposition of the input matrix.
    
    Naive LU decomposition: work column by column, accumulate elementary triangular matrices.
    No pivoting.
    """
    N = a.shape[0]
    
    u = a.copy()
    L = np.eye(N)
    for j in range(N-1):
        lam = np.eye(N)
        gamma = u[j+1:, j] / u[j, j]
        lam[j+1:, j] = -gamma
        #@ is the same as np.dot
        u = lam @ u

        lam[j+1:, j] = gamma
        L = L @ lam
    return L, u


# In[2]:


# Now, generate a full rank matrix and test the naive implementation


N = 6
a = np.zeros((N, N), dtype=float)
for i in range(N):
    for j in range(N):
        a[i, j] = 3. / (0.6*i*j + 1)

np.linalg.matrix_rank(a)


# In[3]:


# Tweak the printing of floating-point numbers, for clarity
np.set_printoptions(precision=3)


# In[4]:


L, u = diy_lu(a)

print(L, "\n")
print(u, "\n")

# Quick sanity check: L times U must equal the original matrix, up to floating-point errors.
print(L@u - a)


# # II. The need for pivoting

# Let's tweak the matrix a little bit, we only change a single element:

# In[6]:


a1 = a.copy()
a1[1, 1] = 3


# Resulting matix still has full rank, but the naive LU routine breaks down.

# In[7]:


np.linalg.matrix_rank(a1)


# In[8]:


#l, u = diy_lu(a1)

#print(l, u)


# ### Test II.1
# 
# For a naive LU decomposition to work, all leading minors of a matrix should be non-zero. Check if this requirement is satisfied for the two matrices `a` and `a1`.
# 
# (20% of the grade)

# In[ ]:
# the function outputs 1 if the requirement for LU decomposition is not satisfied, it outputs 0 if the requirement is satisfied
def check_leading_minors(a):
    not_satisfied=0
    N=a.shape[0]
    for j in range(N):
        dim=np.linalg.matrix_rank(a[:j+1,:j+1])
        if dim<j+1:
            not_satisfied=1
    return not_satisfied

check_leading_minors(a)
check_leading_minors(a1)


# ### Test II.2
# 
# Modify the `diy_lu` routine to implement column pivoting. Keep track of pivots, you can either construct a permutation matrix, or a swap array (your choice).
# 
# (40% of the grade)
#
def diy_lu_pivot(a):
    N = a.shape[0]
    U = a.copy()
    L = np.eye(N)
    P_cum = np.eye(N)
    for j in range(N - 1):
        P_cur=np.eye(N)
        pivot=np.argmax(np.abs(U[j:,j]))
        P_cur[[j,pivot+j]]=P_cur[[pivot+j,j]]
        P_cum=np.dot(P_cur,P_cum)

        U=np.dot(P_cur,U)
        lam = np.eye(N)
        gamma = U[j+1:, j] / U[j, j]
        lam[j+1:, j] = -gamma
        #@ is the same as np.dot
        U = np.dot(lam,U)

        L=np.dot(P_cur,L)
        L = np.dot(lam,L)

    return np.dot(P_cum,np.linalg.inv(L)), U, np.linalg.inv(P_cum)

L,U,P=diy_lu_pivot(a)
L1,U1,P1=diy_lu_pivot(a1)
# Implement a function to reconstruct the original matrix from a decompositon. Test your routines on the matrices `a` and `a1`.
# 
# (40% of the grade)

# In[ ]:

def reconstruct(p,l,u):
    return np.dot(p,np.dot(l,u))
a_rec=reconstruct(P,L,U)
a1_rec=reconstruct(P1,L1,U1)
# ... ENTER YOUR CODE HERE ...

