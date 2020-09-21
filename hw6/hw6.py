#!/usr/bin/env python
# coding: utf-8

# # I. Linear least squares approximation

# Consider a function $y = f(x)$ which is defined by a set of values $y_0, y_1, \cdots, y_n$ at points $x_0, x_1, \cdots, x_n$.

# In[4]:


x = [-1, -0.7, -0.43, -0.14, -0.14, 0.43, 0.71, 1, 1.29, 1.57, 1.86, 2.14, 2.43, 2.71, 3]
y = [-2.25, -0.77, 0.21, 0.44, 0.64, 0.03, -0.22, -0.84, -1.2, -1.03, -0.37, 0.61, 2.67, 5.04, 8.90]


# ### I.I. Find a best fit polynomial
# 
# $$
# P_m(x) = a_0 + a_1 x + \cdots + a_m x^m
# $$
# 
# using the linear least squares approach. To this end
# 
# 1. implement a function which constructs the design matrix using $1, x, \cdots, x^m$ as the basis functions.
# 
# 2. construct explicitly the normal system of equations of the linear least squares problem at fixed $m$.
# 
# 3. Solve the normal equations to find the coefficients of $P_m(x)$ for $m = 0, 1, 2, \dots$. For the linear algebra problem, you can either use library functions (`numpy.linalg.solve`) or your LU factorization code from week 1.
# 
# (20% of the total grade)

# In[10]:


# ... ENTER YOUR CODE HERE
import numpy as np
def poly_fit(x,y,m):
    num_obs=len(x)
    x=np.array(x)
    X=np.zeros([num_obs,m+1])
    y=np.array(y).reshape([-1,1])
    for power in range(m+1):
        X[:,power]=x**power
    return np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
a=poly_fit(x,y,3)
a


# ### I.II 
# 
# To find the optimal value of m, use the following criterion: take $m=0, 1, 2, \dots$, for each value of $m$ compute 
# 
# $$
# \sigma_m^2 = \frac{1}{n - m} \sum_{k=0}^n \left( P_m(x_k) - y_k \right)^2
# $$
# 
# And take the value of $m$, at which $\sigma_m$ stabilizes or starts increasing.
# 
# (20% of the total grade)

def compute_error(x,y,m):
    coefficients=poly_fit(x,y,m)
    num_obs=len(x)
    x=np.array(x)
    X=np.zeros([num_obs,m+1])
    y=np.array(y).reshape([-1,1])
    for power in range(m+1):
        X[:,power]=x**power
    prediction=np.dot(X,coefficients)
    return 1/(num_obs-m)*np.sum((prediction-y)**2)
error=compute_error(x,y,3)


# ... ENTER YOUR CODE HERE ...
import matplotlib.pyplot as plt
plt.plot(x,y,'bo')

for m in [1,2,3,4,5,6,7,8,9]:
    coefficients=poly_fit(x,y,m)
    x=np.arange(-2,4,0.01)
    X=np.zeros([len(x),m+1])
    for power in range(m+1):
        X[:,power]=x**power
    y=np.dot(X,coefficients)
    plt.plot(x,y)

1+1

# Plot your polynomials $P_m(x)$ on one plot, together with the datapoints. Visually compare best-fit polynomials of different degrees. Is the visual comparison consistent with the optimal value of $m$?

# In[ ]:


# ... ENTER YOUR CODE HERE


# ### I.III. Linear least-squares using the QR factorization.
# 
# For the optimal value of $m$ from the previous part, solve the LLS problem using the QR factorization, withou ever forming the normal equations explicitly. For linear algebra, you can use standard library functions (look up `numpy.linalg.solve`, `numpy.linalg.qr` etc) or your code from previous weeks.
# 
# Compare the results with the results of solving the normal system of equations.
# 
# (20% of the grade)

# In[ ]:


# ... ENTER YOUR CODE HERE ...


# # II. Lagrange interpolation

# ### II.1 
# 
# Consider the function, $f(x) = x^2 \cos{x}$. On the interval $x\in [\pi/2, \pi]$, interpolate the function using the Lagrange interpolating polynomial of degree $m$ with $m=1, 2, 3, 4, 5$. Use the uniform mesh. Plot the resulting interpolants together with $f(x)$.
# 
# (20% of the total grade)

# In[ ]:


# ... ENTER YOUR CODE HERE ...


# ### II.2. 
# 
# Repeat the previous task using the Chebyshev nodes. Compare the quality of interpolation on a uniform mesh and Chebyshev nodes for $m=3$.
# 
# (20% of the total grade)

# In[ ]:


# ... ENTER YOUR CODE HERE ...

