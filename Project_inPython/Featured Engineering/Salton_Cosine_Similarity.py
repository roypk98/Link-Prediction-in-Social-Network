#!/usr/bin/env python
# coding: utf-8

# In[1]:


#SAME AS DONE IN "Adamic_Adar_Index"
import pandas as pd
import networkx as nx
import numpy as np
import math


# In[2]:


def salton_cosine(G,feat,ledge):
    feat['Salton_Cosine']=0.0
    for i in feat.index.values:
            feat['Salton_Cosine'][i]=feat['C_Neighbour'][i]/math.sqrt(feat['P_Attachment'][i])


# In[3]:


f_feat=pd.read_csv('FB/FB_Features.csv')
f_ledge=pd.read_csv('FB/FB_LinkedEdges.csv')
G_f=nx.from_pandas_edgelist(f_ledge,'Node_1','Node_2')
salton_cosine(G_f, f_feat, f_ledge)


# In[4]:


ff_feat=pd.read_csv('FB_Food/FBF_Features.csv')
ff_ledge=pd.read_csv('FB_Food/FBF_LinkedEdges.csv')
G_ff=nx.from_pandas_edgelist(ff_ledge,'Node_1','Node_2')
salton_cosine(G_ff, ff_feat, ff_ledge)


# In[5]:


fs_feat=pd.read_csv('FB_Sport/FBS_Features.csv')
fs_ledge=pd.read_csv('FB_Sport/FBS_LinkedEdges.csv')
G_fs=nx.from_pandas_edgelist(fs_ledge,'Node_1','Node_2')
salton_cosine(G_fs, fs_feat, fs_ledge)


# In[6]:


h_feat=pd.read_csv('Ham/Ham_Features.csv')
h_ledge=pd.read_csv('Ham/Ham_LinkedEdges.csv')
G_h=nx.from_pandas_edgelist(h_ledge,'Node_1','Node_2')
salton_cosine(G_h, h_feat, h_ledge)


# In[7]:


v_feat=pd.read_csv('Vote/Vote_Features.csv')
v_ledge=pd.read_csv('Vote/Vote_LinkedEdges.csv')
G_v=nx.from_pandas_edgelist(v_ledge,'Node_1','Node_2')
salton_cosine(G_v, v_feat, v_ledge)


# In[8]:


f_feat.to_csv(r'FB/FB_Features.csv',index=False)


# In[9]:


ff_feat.to_csv(r'FB_Food/FBF_Features.csv',index=False)


# In[10]:


fs_feat.to_csv(r'FB_Sport/FBS_Features.csv',index=False)


# In[11]:


h_feat.to_csv(r'Ham/Ham_Features.csv',index=False)


# In[12]:


v_feat.to_csv(r'Vote/Vote_Features.csv',index=False)

