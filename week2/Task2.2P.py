#!/usr/bin/env python
# coding: utf-8

# In[2]:


def recur_factorial(x):   # function for recursive factorial
    if x==1:
        return x
    else:
        return x*recur_factorial(x-1)
n = int(input("Enter a number: ")) # Ask the user to input a number
while n < 0:
    n = int(input("Please enter a nonnegative integer: "))
if n==0:
    print("Factorial of 0:\n0")
else:
    print("Factorial of "+str(n)+":\n"+str(recur_factorial(n)))


# In[ ]:




