#!/usr/bin/env python
# coding: utf-8

# In[200]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[201]:


df = pd.read_csv("Malicious_or_criminal_attacks_breakdown-Top_five_industry_sectors_July-Dec-2019.csv", engine="python",index_col=0,encoding="unicode_escape")


# In[202]:


data = np.array([df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3],df.iloc[:,4]])


# In[203]:




colors = ['Red', 'Yellow', 'Blue', 'Green']
labels = ['Cyber incident', 'Theft of paperwork or data storage device', 'Rogue employee / insider threat','Social engineering/ impersonation']
x_pos = np.arange(5)
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14, 5), dpi=100)

cyber_incident = data[:,0] # cyber incident
theft_data_storage = data[:,1]  # Theft of paperwork or data storage device
rogue_employee = data[:,2]     # Rogue employee / insider threat
social_engineering = data[:,3]  # Social engineering/ impersonation

w1 = 0.2
w2=0.4
for i in np.arange(len(labels)):
    ax[0].bar(x_pos + i*w1, data[:,i], width=w1, color=colors[i], label=labels[i], align='center')

    

ax[1].bar(x_pos, data[:,0], width=w2, align='center', label=labels[0], color=colors[0])
ax[1].bar(x_pos, data[:,1], width=w2, align='center', label=labels[1], color=colors[1], bottom=cyber_incident)
ax[1].bar(x_pos, data[:,2], width=w2, align='center', label=labels[2], color=colors[2], bottom = cyber_incident+theft_data_storage)
ax[1].bar(x_pos, data[:,3], width=w2, align='center', label=labels[3], color=colors[3], bottom=cyber_incident+theft_data_storage+rogue_employee)


ax[0].set_xlabel("The top five industry sectors")
ax[0].set_ylabel("Number of attack")
ax[0].set_title("Type of attack by top five industry sectors")
ax[1].set_xlabel("The top five industry sectors")
ax[1].set_ylabel("Number of attack")
ax[1].set_title("Type of attack by top five industry sectors")

ax[0].set_xticks(x_pos+w1/2)
ax[0].set_xticklabels(df.columns,rotation=90)
ax[1].set_xticks(x_pos)
ax[1].set_xticklabels(df.columns, rotation=90)

ax[0].legend()
ax[1].legend(loc=0)


# In[ ]:




