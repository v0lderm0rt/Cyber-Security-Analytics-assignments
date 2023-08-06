#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import factorial


# In[2]:


def factCalc(x):
     while x < 0:
            x=int(input("Please enter a nonnegative integer: "))
        
        
     print("Factorial of "+ str(x)+" is "+ str(factorial(x)))
       
        


# In[3]:


num = int(input("Enter a number: "))
factCalc(num)


# In[ ]:





# In[ ]:




