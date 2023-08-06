#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.DataFrame({'a': np.random.randint(0, 50, size=100)})
df['b'] = df['a'] + np.random.normal(0, 10, size=100)
df['c'] = 100 - 2 * df['a'] + np.random.normal(0, 10, size=100)
df['d'] = np.random.randint(0, 50, 100)


# In[17]:


# Pearson's-r coefficient and corrcoef() for df['a'] and df['b']
print("Pearson's r coefficient for df[‘a’] and df[‘b’] is ",scipy.stats.pearsonr(df['a'], df['b'])[0])
print("corrcoef() for df['a'] and df['b']: ", np.corrcoef(df['a'], df['b']))
# Pearson's-r coefficient and corrcoef for df['a'] and df['c']
print("Pearson's-r coefficient for df['a'] and df['c']: ", scipy.stats.pearsonr(df['a'], df['c'])[0])
print("corrcoef() for df['a'] and df['c']: ", np.corrcoef(df['a'], df['c']))

# Pearson's-r coefficient and corrcoef for df['a'] and df['d']
print("Pearson's-r coefficient for df['a'] and df['d']: ", scipy.stats.pearsonr(df['a'], df['d'])[0])
print("corrcoef() for df['a'] and df['d']: ", np.corrcoef(df['a'], df['d']))


# In[20]:


# to visualize data with positive correlation
x = df['a']
y = df['b']
plt.subplots(figsize=(7, 5), dpi=100)
line_coef = np.polyfit(x, y, 1)
xx = np.arange(0, 50, 0.1)
yy = line_coef[0]*xx + line_coef[1]
plt.scatter(x, y, color='red')
plt.plot(xx, yy, color='blue', lw=2)
plt.show()


# In[21]:


# to visualize data with negative correlation
x = df['a']
y = df['c']
plt.subplots(figsize=(7, 5), dpi=100)
line_coef = np.polyfit(x, y, 1)
xx = np.arange(0, 50, 0.1)
yy = line_coef[0]*xx + line_coef[1]
plt.scatter(x, y, color='red')
plt.plot(xx, yy, color='blue', lw=2)
plt.show()


# In[ ]:




