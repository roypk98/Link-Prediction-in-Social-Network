#!/usr/bin/env python
# coding: utf-8

# # DO RUN

# In[1]:


#imp libraries
import pandas as pd
import networkx as nx
import numpy as np
import math


# In[2]:


#defined function for the algorithm to be applied
def adamic_adar(G,feat,ledge):
    #adding column for thr algorithm value to store for every edge initialised as 0.0
    feat['Adamic_Adar']=0.0
    #running for all edges in missing edges
    for i in feat.index.values:
        #checking for the presence of node in the graph of linked edges
        if(G.has_node(feat['Node_1'][i]) and G.has_node(feat['Node_2'][i])):
            try:
                values=nx.adamic_adar_index(G,[(feat['Node_1'][i],feat['Node_2'][i])])
                for v in values:
                    feat['Adamic_Adar'][i]=v[2]
            except ZeroDivisionError:#to avoid zero division error
                continue


# In[3]:


#applied algorithm on features(changed name for missing edges) and created graph of linked edges
#same for In[3] to In[7]
f_feat=pd.read_csv('FB/FB_Features.csv')
f_ledge=pd.read_csv('FB/FB_LinkedEdges.csv')
G_f=nx.from_pandas_edgelist(f_ledge,'Node_1','Node_2')
adamic_adar(G_f, f_feat, f_ledge)


# In[4]:


ff_feat=pd.read_csv('FB_Food/FBF_Features.csv')
ff_ledge=pd.read_csv('FB_Food/FBF_LinkedEdges.csv')
G_ff=nx.from_pandas_edgelist(ff_ledge,'Node_1','Node_2')
adamic_adar(G_ff, ff_feat, ff_ledge)


# In[5]:


fs_feat=pd.read_csv('FB_Sport/FBS_Features.csv')
fs_ledge=pd.read_csv('FB_Sport/FBS_LinkedEdges.csv')
G_fs=nx.from_pandas_edgelist(fs_ledge,'Node_1','Node_2')
adamic_adar(G_fs, fs_feat, fs_ledge)


# In[6]:


h_feat=pd.read_csv('Ham/Ham_Features.csv')
h_ledge=pd.read_csv('Ham/Ham_LinkedEdges.csv')
G_h=nx.from_pandas_edgelist(h_ledge,'Node_1','Node_2')
adamic_adar(G_h, h_feat, h_ledge)


# In[7]:


v_feat=pd.read_csv('Vote/Vote_Features.csv')
v_ledge=pd.read_csv('Vote/Vote_LinkedEdges.csv')
G_v=nx.from_pandas_edgelist(v_ledge,'Node_1','Node_2')
adamic_adar(G_v, v_feat, v_ledge)


# In[8]:


#converted to .csv file to make changes permanent
#same for In[8] to In[12]
f_feat.to_csv(r'FB/FB_Features.csv',index=False)


# In[9]:


ff_feat.to_csv(r'FB_Food/FBF_Features.csv',index=False)


# In[10]:


fs_feat.to_csv(r'FB_Sport/FBS_Features.csv',index=False)


# In[11]:


h_feat.to_csv(r'Ham/Ham_Features.csv',index=False)


# In[12]:


v_feat.to_csv(r'Vote/Vote_Features.csv',index=False)

