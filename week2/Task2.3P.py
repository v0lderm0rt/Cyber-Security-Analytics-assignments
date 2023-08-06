#!/usr/bin/env python
# coding: utf-8

# In[3]:


import re


# In[4]:


text = "email: shang.gao@test2.server.com, usename: shang.gao, host: test2.server.com"


# In[44]:


enter_email = input("Please input your email address:")

x = re.search(enter_email,text)
while not x:
    print("Not a valid email")
    enter_email = input("Please input your email address:")

    x = re.search(enter_email,text)
if x:
    print(text)


# In[ ]:




