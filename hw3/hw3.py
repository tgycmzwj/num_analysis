#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

from numpy.testing import assert_allclose



def householder_new(vec):
    vec = np.asarray(vec, dtype=float)
    if vec.ndim != 1:
        raise ValueError("vec.ndim = %s, expected 1" % vec.ndim)
    u=vec.copy()
    u[0]=-np.sum(vec[1:]**2)/(vec[0]+np.linalg.norm(vec,ord=2))
    u=u/np.linalg.norm(u,ord=2)
    return u
def qr_decomp_new(a):
    a1 = np.array(a, copy=True, dtype=float)
    m, n = a1.shape
    Q = []
    # ... ENTER YOUR CODE HERE ...
    for col in range(n):
        v=householder_new(a1[col:,col])
        Q.append(np.concatenate((np.zeros(col),v)))
        a1=a1-2*Q[-1].reshape(-1,1) @ (Q[-1].reshape(1,-1) @ a1)
    Q=np.array(Q)
    return Q,a1
rndm = np.random.RandomState(1234)
a = rndm.uniform(size=(5, 3))
qrr = qr_decomp_new(a)

