#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


colors = ['blue','red','green','yellow']
labels = ["Cyber incident", "Theft of paperwork or data storage device", "Rogue employee", "Social engineering / impersonation"]
no_of_attacks = [108,32,26,11]
fig, ax = plt.subplots(figsize=(7,5),dpi=100)
x_pos = np.arange(len(labels))
ax.bar(x_pos, no_of_attacks, align='center', color=colors)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=90)
ax.set_xlabel("Attack type")
ax.set_ylabel("Number of attacks per attack type")
ax.set_title(" Number of malicious or criminal attack July-December-2019")


# In[5]:


labels_pie = ["Health service providers", "Finance", "Education", "Legal,accounting & management services", "Personal services"]


# In[10]:


freq = [63,40,30,30,14]
fig, ax = plt.subplots(figsize=(10,10))
ax.pie(freq, labels=labels_pie,autopct="%1.1f")


# In[ ]:




