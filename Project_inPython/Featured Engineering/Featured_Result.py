#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imp libraries
import pandas as pd


# In[2]:


#loaded .csv file and display the values of all features
#same In[2] to In[6]
fb=pd.read_csv('FB/FB_Features.csv')
fb


# In[3]:


fbf=pd.read_csv('FB_Food/FBF_Features.csv')
fbf


# In[4]:


fbs=pd.read_csv('FB_Sport/FBS_Features.csv')
fbs


# In[5]:


ham=pd.read_csv('Ham/Ham_Features.csv')
ham


# In[6]:


vote=pd.read_csv('Vote/Vote_Features.csv')
vote

