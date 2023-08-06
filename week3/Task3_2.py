#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[61]:


df = pd.read_csv(r'result_withoutTotal.csv') 


# In[5]:


ass1_3_5per = df.loc[:,['Ass1','Ass3']].sum(axis='columns')*5/100 # (Ass1+Ass3)*5%


# In[6]:


ass2_4_15per = df.loc[:,['Ass2','Ass4']].sum(axis='columns')*15/100 # (Ass1+Ass4)*15%


# In[7]:


exam_60_per = df.loc[:,['Exam']].sum(axis='columns') * 60/100 # 60% of exam


# In[8]:


Total = ass1_3_5per + ass2_4_15per + exam_60_per # Total score


# In[9]:


df['Total']=Total.clip(upper=100) # add the Total values to the dataframe and set maximum value as 100


# In[23]:


df['Final'] = Total.apply(np.round) # round off 'Total' values to nearest integer and adding them as a new column 'Final'


# In[ ]:





# In[40]:


grade_col = []                    # Creating a list to assign grades
for i in df['Final']:
    if i <= 49.45:
        grade_col.append('N')
    elif i > 49.45 and i <= 59.45:
        grade_col.append('P')
    elif i > 59.45 and i <= 69.45:
        grade_col.append('C')
    elif i > 69.45 and i <= 79.45:
        grade_col.append('D')
    elif i > 79.45:
        grade_col.append('HD')


df['Grade'] = grade_col # creating the 'Grade' column using the list


# In[50]:


df_2 = df.loc[(df['Exam'] < 48)] # creating a dataframe for students who got < 48 for exam


# In[53]:


df.to_csv('result_updated.csv')  # the result data file with the 3 new columns to a file called result_updated.csv.


# In[56]:


df_2.to_csv('failedhurdle.csv') # the students’ records with exam score < 48 to a file called failedhurdle.csv


# In[58]:


print(df)


# In[59]:


print(df_2) # students with exam < 48


# In[60]:


print(df.loc[(df['Exam'] > 100)]) # the students with exam score > 100


# In[ ]:




