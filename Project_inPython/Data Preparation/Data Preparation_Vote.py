#!/usr/bin/env python
# coding: utf-8

# # DO NOT RUN
# ### It will effect the result

# In[1]:


#SAME AS DONE IN "Data Preparation_FB"
import pandas as pd
import networkx as nx
import numpy as np


# In[2]:


with open("Vote/email-Eu-core.txt") as f:
    f_links = f.read().splitlines() 


# In[3]:


len(f_links)


# In[4]:


node_list_1 = []
node_list_2 = []

for i in f_links:
    node_list_1.append(i.split(',')[1])
    node_list_2.append(i.split(',')[0])

f_df = pd.DataFrame({'Node_1': node_list_1, 'Node_2': node_list_2})


# In[5]:


f_df


# In[6]:


f_df.to_csv(r'Vote/Vote.csv',index=False)


# In[7]:


G = nx.from_pandas_edgelist(f_df, "Node_1", "Node_2")


# In[8]:


node_list = node_list_1 + node_list_2
node_list = list(dict.fromkeys(node_list))
adj_G = nx.to_numpy_matrix(G, nodelist = node_list)


# In[9]:


adj_G.shape


# In[10]:


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


len(all_unconnected_pairs)


# In[12]:


node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]

f_data = pd.DataFrame({'Node_1':node_1_unlinked, 
                     'Node_2':node_2_unlinked})

f_data['Link'] = 0


# In[13]:


f_data


# In[14]:


omissible_links_index=[]
f_df_temp=f_df.copy()
for i in np.random.choice(f_df.index,2557, replace=False):
    G_temp=nx.from_pandas_edgelist(f_df_temp.drop(index=i), "Node_1", "Node_2")
    if(G_temp.number_of_nodes()==len(node_list)):
        omissible_links_index.append(i)
        f_df_temp = f_df_temp.drop(index = i)


# In[15]:


len(omissible_links_index)


# In[16]:


f_df_ghost = f_df.loc[omissible_links_index]
f_df_ghost['Link'] = 1
f_data = f_data.append(f_df_ghost[['Node_1', 'Node_2', 'Link']], ignore_index=True)


# In[17]:


f_data['Link'].value_counts()


# In[18]:


f_data.sort_values(by='Node_1', inplace=True)


# In[19]:


f_data


# In[20]:


f_data.to_csv(r'Vote/Vote_MissingEdges.csv',index=False)


# In[21]:


f_df_temp


# In[22]:


f_df_temp.to_csv(r'Vote/Vote_LinkedEdges.csv',index=False)

