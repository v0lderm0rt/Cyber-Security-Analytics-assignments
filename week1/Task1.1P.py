#!/usr/bin/python


# In[13]:

 # Days of the week as a list,tuple,set and dictionary
my_list = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

my_tuple = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
my_set = {"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"}
my_dict = {1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday",7:"Sunday"}


# In[16]:

# print the type and the days of the week
print(type(my_list))
for i in my_list:
    print(i)
print(type(my_tuple))
for i in my_tuple:
    print(i)
print(type(my_set))
for i in my_set:
    print(i)
print(type(my_dict))
for value in my_dict.values():
    print(value)


# In[ ]:




