#!/usr/bin/env python
# coding: utf-8

# # DO NOT RUN
# ### It will effect the result

# In[1]:


#all important libraries
import pandas as pd
import networkx as nx
import numpy as np


# In[2]:


#extracting data from file
with open("FB/socfb-Haverford76.mtx") as f:
    f_links = f.read().splitlines() 


# In[3]:


#nuber of edges
len(f_links)


# In[4]:


#creating DataFrame f_df
node_list_1 = []
node_list_2 = []

for i in f_links:
    node_list_1.append(i.split(',')[1])
    node_list_2.append(i.split(',')[0])

f_df = pd.DataFrame({'Node_1': node_list_1, 'Node_2': node_list_2})


# In[5]:


#actual data downloaded from net 
f_df


# In[6]:


#converted to .csv format
f_df.to_csv(r'FB/FB.csv',index=False)


# In[7]:


#converted f_df to graph
G = nx.from_pandas_edgelist(f_df, "Node_1", "Node_2")


# In[8]:


#creating matrix
node_list = node_list_1 + node_list_2
node_list = list(dict.fromkeys(node_list))
adj_G = nx.to_numpy_matrix(G, nodelist = node_list)


# In[9]:


#matrix of number of nodes X number of nodes
adj_G.shape


# In[10]:


#extracting few missing edges from all missing edges of the graph with condition shortest path = 1 
all_unconnected_pairs = []

offset = 0
for i in range(0,adj_G.shape[0]):
    for j in range(offset,adj_G.shape[1]):
        if i != j:
            try:
                shortest_path=nx.shortest_path_length(G, str(i), str(j))
                if  shortest_path==1:
                    if adj_G[i,j] == 0:
                        all_unconnected_pairs.append([node_list[i],node_list[j]])
            except nx.NodeNotFound:
                continue
            except nx.NetworkXNoPath:
                continue
    offset = offset + 1


# In[11]:


#extracted 56054 missing edges
len(all_unconnected_pairs)


# In[12]:


#creating dataframe of all missing edges extracted
node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]

f_data = pd.DataFrame({'Node_1':node_1_unlinked, 
                     'Node_2':node_2_unlinked})

f_data['Link'] = 0


# In[13]:


#dataframe of all missing edges
f_data


# In[14]:


#deleting almost 10% edges from the actual data without reducing number of nodes
omissible_links_index=[]
f_df_temp=f_df.copy()
for i in np.random.choice(f_df.index,5959, replace=False):
    G_temp=nx.from_pandas_edgelist(f_df_temp.drop(index=i), "Node_1", "Node_2")
    if(G_temp.number_of_nodes()==len(node_list)):
        omissible_links_index.append(i)
        f_df_temp = f_df_temp.drop(index = i)


# In[15]:


#length of deleted edges
len(omissible_links_index)


# In[16]:


#adding to the dataframe of missing edges
f_df_ghost = f_df.loc[omissible_links_index]
f_df_ghost['Link'] = 1
f_data = f_data.append(f_df_ghost[['Node_1', 'Node_2', 'Link']], ignore_index=True)


# In[17]:


# 0 means no link in future and 1 means link in future
f_data['Link'].value_counts()


# In[18]:


#for organising the data
f_data.sort_values(by='Node_1', inplace=True)


# In[19]:


#dataframe having missing edges extracted from above algorithm and deleted edges from the actual data
f_data


# In[20]:


#converted to .csv file
#all the prediction algorithm and featured engineering will be applied on this file
f_data.to_csv(r'FB/FB_MissingEdges.csv',index=False)


# In[21]:


#actual data after deleting few edges
f_df_temp


# In[22]:


#converted to .csv file
#now this will act as actual data 
f_df_temp.to_csv(r'FB/FB_LinkedEdges.csv',index=False)

