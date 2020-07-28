#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imp libraries
import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[4]:


# function for neural network
def neural_network(feat):
    
    X=feat.drop(['Link', 'C_Neighbour', 'P_Attachment',
       'Jaccard_Index', 'Adamic_Adar', 'R_Allocation', 'Salton_Cosine',
       'Sorensen_Index', 'Promoted_Index', 'LHN_Index'],axis=1)
    y=feat['Link']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    neural=MLPClassifier()
    neural.fit(X_train,y_train)
    predictions=neural.predict(X_test)
    
    print('Confusion Matrix\n')
    print(confusion_matrix(y_test,predictions))
    print('\nClassification Report\n')
    print(classification_report(y_test,predictions))
    
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, predictions)
    print('ROC_AUC_Score', roc_auc_score(y_test, predictions))
    
    plt.subplots(1, figsize=(5,5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[5]:


#function for knearest_neighbor
def knearest_neighbour(feat_temp):
    
    scaler=StandardScaler()
    scaler.fit(feat_temp.drop('Link',axis=1))
    scaled_feat=scaler.transform(feat_temp.drop('Link',axis=1))
    
    feat=pd.DataFrame(scaled_feat,columns=feat_temp.columns.drop('Link'))
    
    X=feat.drop(['C_Neighbour', 'P_Attachment', 'Jaccard_Index', 'Adamic_Adar', 'R_Allocation', 'Salton_Cosine',
       'Sorensen_Index', 'Promoted_Index', 'LHN_Index'],axis=1)
    y=feat_temp['Link']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    knn=KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train,y_train)
    predictions=knn.predict(X_test)
        
    print('Confusion Matrix\n')
    print(confusion_matrix(y_test,predictions))
    print('\nClassification Report\n')
    print(classification_report(y_test,predictions))
    
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, predictions)
    print('ROC_AUC_Score', roc_auc_score(y_test, predictions))
    
    plt.subplots(1, figsize=(5,5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[6]:


#function for random_forest
def random_forest(feat):
    
    X=feat.drop(['Link', 'C_Neighbour', 'P_Attachment',
       'Jaccard_Index', 'Adamic_Adar', 'R_Allocation', 'Salton_Cosine',
       'Sorensen_Index', 'Promoted_Index', 'LHN_Index'],axis=1)
    y=feat['Link']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    rfc=RandomForestClassifier()
    rfc.fit(X_train,y_train)
    predictions=rfc.predict(X_test)
    
    print('Confusion Matrix\n')
    print(confusion_matrix(y_test,predictions))
    print('\nClassification Report\n')
    print(classification_report(y_test,predictions))
    
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, predictions)
    print('ROC_AUC_Score', roc_auc_score(y_test, predictions))
    
    plt.subplots(1, figsize=(5,5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[7]:


#applying neural network for all 5 dataset
for i in range(5):
    if(i==0):
        f_feat=pd.read_csv('FB/FB_Features.csv')
        print('FB_Prediction\n')
        neural_network(f_feat)
    elif(i==1):
        ff_feat=pd.read_csv('FB_Food/FBF_Features.csv')
        print('FB_Food_Prediction\n')
        neural_network(ff_feat)
    elif(i==2):
        fs_feat=pd.read_csv('FB_Sport/FBS_Features.csv')
        print('FB_Sport_Prediction\n')
        neural_network(fs_feat)
    elif(i==3):
        h_feat=pd.read_csv('Ham/Ham_Features.csv')
        print('Hamsterster_Prediction\n')
        neural_network(h_feat)
    else:
        v_feat=pd.read_csv('Vote/Vote_Features.csv')
        print('Vote_Prediction\n')
        neural_network(v_feat)


# In[8]:


#applying knearest neighbors for all 5 dataset
for i in range(5):
    if(i==0):
        f_feat=pd.read_csv('FB/FB_Features.csv')
        print('FB_Prediction\n')
        knearest_neighbour(f_feat)
    elif(i==1):
        ff_feat=pd.read_csv('FB_Food/FBF_Features.csv')
        print('FB_Food_Prediction\n')
        knearest_neighbour(ff_feat)
    elif(i==2):
        fs_feat=pd.read_csv('FB_Sport/FBS_Features.csv')
        print('FB_Sport_Prediction\n')
        knearest_neighbour(fs_feat)
    elif(i==3):
        h_feat=pd.read_csv('Ham/Ham_Features.csv')
        print('Hamsterster_Prediction\n')
        knearest_neighbour(h_feat)
    else:
        v_feat=pd.read_csv('Vote/Vote_Features.csv')
        print('Vote_Prediction\n')
        knearest_neighbour(v_feat)


# In[9]:


#applying random forest for all 5 dataset
for i in range(5):
    if(i==0):
        f_feat=pd.read_csv('FB/FB_Features.csv')
        print('FB_Prediction\n')
        random_forest(f_feat)
    elif(i==1):
        ff_feat=pd.read_csv('FB_Food/FBF_Features.csv')
        print('FB_Food_Prediction\n')
        random_forest(ff_feat)
    elif(i==2):
        fs_feat=pd.read_csv('FB_Sport/FBS_Features.csv')
        print('FB_Sport_Prediction\n')
        random_forest(fs_feat)
    elif(i==3):
        h_feat=pd.read_csv('Ham/Ham_Features.csv')
        print('Hamsterster_Prediction\n')
        random_forest(h_feat)
    else:
        v_feat=pd.read_csv('Vote/Vote_Features.csv')
        print('Vote_Prediction\n')
        random_forest(v_feat)

