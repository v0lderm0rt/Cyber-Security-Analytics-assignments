#!/usr/bin/env python
# coding: utf-8

# In[51]:


width = int(input("Enter your width: "))    # Enter width
while width <= 0:
    print("Width cannot be less than or equal to zero")   # Width cannot be less than or equal to zero
    width = int(input("Enter your width: "))
    
height = int(input("Enter your height: "))   # Enter your height
while height <= 0:
    print("Height cannot be less than or equal to zero") # Height cannot be less than or equal to zero
    height = int(input("Enter your height: "))


# In[52]:


for i in range(0,height):
    for j in range(0,width+1):
        if j == width:
            print(j*"* ")
        
    
        
        


# In[ ]:




