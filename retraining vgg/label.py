
# coding: utf-8

# In[10]:


import os
import shutil


# In[11]:


# setting file path
parent = os.getcwd()
datapath = os.path.join(parent,'Dataset','UTK','UTKFace') 
datalist = os.listdir(datapath)


# In[12]:


datapath


# In[13]:


len(datalist)


# # Labelling 1-116

# In[14]:


# create new path
new_path = os.path.join(parent ,'Dataset','Labeled')
if not os.path.isdir(new_path):
    try:
        original_umask = os.umask(0)
        os.makedirs(new_path, 0755)
    finally:
        os.umask(original_umask)


# In[21]:


for d in datalist:
    
    # Create subfolder using label values
    label = "{0:03}".format(int(d.split('_')[0]))
    path = os.path.join(new_path,label)
    if not os.path.isdir(path):
        try:
            original_umask = os.umask(0)
            os.makedirs(path, 0755)
        finally:
            os.umask(original_umask)
    
    # Move file to new subfolders
    shutil.copy(os.path.join(datapath,d), path)


# # Labeling every five years old

# In[24]:


# create new path
new_path = os.path.join(parent ,'Dataset','LabelBy5')
if not os.path.isdir(new_path):
    try:
        original_umask = os.umask(0)
        os.makedirs(new_path, 0755)
    finally:
        os.umask(original_umask)


# In[ ]:


for d in datalist:
    
    # Create subfolder using label values
    label =  str(int(int(d.split('_')[0])/5))
    path = os.path.join(new_path,label)
    if not os.path.isdir(path):
        try:
            original_umask = os.umask(0)
            os.makedirs(path, 0755)
        finally:
            os.umask(original_umask)
    
    # Move file to new subfolders
    shutil.copy(os.path.join(datapath,d), path)


# # Label with Gender

# In[8]:


# create new path
new_path = os.path.join(parent ,'Dataset','LabelGender')
if not os.path.isdir(new_path):
    try:
        original_umask = os.umask(0)
        os.makedirs(new_path, 0755)
    finally:
        os.umask(original_umask)


# In[9]:


for d in datalist:
    
    # Create subfolder using label values
    label =  d.split('_')[1]
    path = os.path.join(new_path,label)
    if not os.path.isdir(path):
        try:
            original_umask = os.umask(0)
            os.makedirs(path, 0755)
        finally:
            os.umask(original_umask)
    
    # Move file to new subfolders
    shutil.copy(os.path.join(datapath,d), path)


# # Label with Race

# In[10]:


# create new path
new_path = os.path.join(parent ,'Dataset','LabelRace')
if not os.path.isdir(new_path):
    try:
        original_umask = os.umask(0)
        os.makedirs(new_path, 0755)
    finally:
        os.umask(original_umask)


# In[11]:


for d in datalist:
    
    # Create subfolder using label values
    label =  d.split('_')[2]
    path = os.path.join(new_path,label)
    if not os.path.isdir(path):
        try:
            original_umask = os.umask(0)
            os.makedirs(path, 0755)
        finally:
            os.umask(original_umask)
    
    # Move file to new subfolders
    shutil.copy(os.path.join(datapath,d), path)

