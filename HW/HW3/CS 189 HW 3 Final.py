#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.io import loadmat
import pandas as pd
import datetime


# ## Problem 3

# In[10]:


#Part 1. 


mu = np.array([1, 1])
cov = np.array([[1, 0], [0, 2]])
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
z = multivariate_normal.pdf(pos, mean=mu, cov=cov)
fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='Density')
 
plt.show()


# In[11]:


#Part 2


mu = np.array([-1, 2])
cov = np.array([[2, 1], [1, 4]])
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
z = multivariate_normal.pdf(pos, mean=mu, cov=cov)
fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='Density')
plt.title('Isocontours of f(μ,Σ) with μ = [-1, 2] and Σ = [[2, 1], [1, 4]]')
 
plt.show()




# In[12]:


# Part 3


mu1 = np.array([0, 2])
mu2 = np.array([2, 0])
cov = np.array([[2, 1], [1, 1]])


x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y


z = multivariate_normal.pdf(pos, mean=mu1, cov=cov) - multivariate_normal.pdf(pos, mean=mu2, cov=cov)

fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='f(µ1, Σ1) − f(µ2, Σ2)')
plt.title('Isocontours of f(µ1, Σ1) − f(µ2, Σ2)')
 
plt.show()






# In[13]:


# Part 4

mu1 = np.array([0, 2])
mu2 = np.array([2, 0])
cov1 = np.array([[2, 1], [1, 4]])
cov2 = np.array([[2, 1], [1, 4]])

x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y


z = multivariate_normal.pdf(pos, mean=mu1, cov=cov1) - multivariate_normal.pdf(pos, mean=mu2, cov=cov2)


fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='Density')

plt.show()



# In[14]:


# Part 5 



mu1 = np.array([1, 1])
mu2 = np.array([-1, -1])
cov1 = np.array([[2, 0], [0, 1]])
cov2 = np.array([[2, 1], [1, 2]])


x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y


z1 = multivariate_normal.pdf(pos, mean=mu1, cov=cov1)
z2 = multivariate_normal.pdf(pos, mean=mu2, cov=cov2)


z = z1 - z2


fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='Density')

plt.show()





# ## Problem 4

# In[15]:


np.random.seed(42)


mean = np.array([3, 3.5])
cov = np.array([[9, 4.5], [4.5, 4]])


n = 100
samples = np.random.multivariate_normal(mean, cov, size=n)
sample_mean = np.mean(samples, axis=0)
sample_cov = np.cov(samples.T)
eigenvalues, eigenvectors = np.linalg.eig(sample_cov)


sort_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sort_indices]
eigenvalues = eigenvalues[sort_indices]


plt.scatter(samples[:, 0], samples[:, 1])
plt.arrow(sample_mean[0], sample_mean[1], eigenvalues[0] * eigenvectors[0, 0], eigenvalues[0] * eigenvectors[1, 0], width=0.1, color='r')
plt.arrow(sample_mean[0], sample_mean[1], eigenvalues[1] * eigenvectors[0, 1], eigenvalues[1] * eigenvectors[1, 1], width=0.1, color='r')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter plot of samples and covariance eigenvectors')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#ROTATED
rotated_samples = np.dot(eigenvectors.T, (samples - sample_mean).T).T
plt.scatter(rotated_samples[:, 0], rotated_samples[:, 1])
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter plot of rotated samples')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

