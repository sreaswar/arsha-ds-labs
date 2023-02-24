#!/usr/bin/env python
# coding: utf-8

# In[33]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[12]:


import numpy as np
from numpy.random import seed
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint
np.__name__


# In[21]:


from np.random import seed


# In[11]:


rand(10)
randint(0,10,15)
randn(10)


# In[24]:


np.random.seed(0)
m1 = np.random.randint(0,10, (4,4))
print(m1)
m1[0][2]


# In[26]:


np.random.seed(0)
m2 = np.random.randint(0,10, (4,4))
print(m2)
m2[0][2]


# In[44]:


import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics as st
mu, sigma = 10, 1 # mean and standard deviation
d1 = np.random.normal(mu, sigma, 1000)
#print(d1)
mean = st.mean(d1)
sd = st.stdev(d1)
print(mean,sd)
count, bins, ignored = plt.hist(d1, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()


# In[46]:


import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics as st
a, m = 3., 2.  # shape and mode
d1 = (np.random.pareto(a, 1000) + 1) * m

#print(d1)
mean = st.mean(d1)
sd = st.stdev(d1)
print(mean,sd)
count, bins, ignored = plt.hist(d1, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()


# In[ ]:




