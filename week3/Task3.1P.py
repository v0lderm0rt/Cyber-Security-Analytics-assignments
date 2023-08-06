#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[33]:


df = pd.read_csv(r'result.csv')


# In[34]:


# mean,min,max of Ass1
avg_ass1 = df['Ass1'].mean()
min_ass1 = df['Ass1'].min()
max_ass1 = df['Ass1'].max()

# mean,min,max of Ass2
avg_ass2 = df['Ass2'].mean()
min_ass2 = df['Ass2'].min()
max_ass2 = df['Ass2'].max()

# mean,min,max of Ass3
avg_ass3 = df['Ass3'].mean()
min_ass3 = df['Ass3'].min()
max_ass3 = df['Ass3'].max()

# mean,min,max of Ass4
avg_ass4 = df['Ass4'].mean()
min_ass4 = df['Ass4'].min()
max_ass4 = df['Ass4'].max()

# highest total value
highest_total = df['Total'].max()


# In[37]:


print("ass1 average:" + str(avg_ass1) + "\nass1 min:" + str(min_ass1) + "\nass1 max:" + str(max_ass1))
print("\nass2 average:" + str(avg_ass2) + "\nass2 min:" + str(min_ass2) + "\nass2 max:" + str(max_ass2))
print("\nass3 average:" + str(avg_ass3) + "\nass3 min:" + str(min_ass3) + "\nass3 max:" + str(max_ass3) + "\n")
print("Student with highest total:")
print(df.loc[df['Total'] == highest_total])


# In[ ]:




