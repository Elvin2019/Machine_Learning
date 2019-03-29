#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:45:12 2019

@author: HP
"""
import pandas as pd
import numpy as np


# In[2]:


#first of all we need load data therefore we should clean unordered data
#this function labels each data based on '#' - hash each hash has own features and labels

def Load_Data_Function(filename):
    data = pd.DataFrame(columns = ['L_stands','Values']) #loading data into dataframe
    with open(filename) as txtFile: #opening files
        for each, stand in enumerate(txtFile): #
            L_stands, Values = stand.split('#')
            Values = [float(V.strip()) for V in Values.split(';')]
            data.loc[each] = [L_stands, Values]
            each = each + 1
    data.iloc[:,0] = data.iloc[:,0].astype(np.int)
    return data           


# # LOADING DATA + 

# In[3]:


data_train = Load_Data_Function('Train.txt')
data_test = Load_Data_Function('Test.txt')
data_cv = Load_Data_Function('Cross_Validation.txt')


# In[4]:


def List_Function(data):
    return np.array(data.values.tolist())

def Values_Function(data):
    return data.values


# In[5]:


#loading features of each data to variables
data_train_X = List_Function(data_train.iloc[:,1])
data_test_X = List_Function(data_test.iloc[:,1])
data_cv_X = List_Function(data_cv.iloc[:,1])

#loading labels of each data to variables
data_train_y = Values_Function(data_train.iloc[:,0])
data_test_y = Values_Function(data_test.iloc[:,0])
data_cv_y = Values_Function(data_cv.iloc[:,0])


# # NEURAL NETWORKS + 

# In[6]:


from sklearn.neural_network import MLPClassifier as NT
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[7]:


MLP_CLF = NT(hidden_layer_sizes=(100,100,100), alpha=0.0001) 
MLP_CLF.fit(data_train_X, data_train_y) #finding coefficients 
#prediction 
y_pred = MLP_CLF.predict(data_test_X)
#finding accurrracy percentage
MLP_CLF_AC_1 = round(accuracy_score(data_test_y, y_pred),4)
print('Evaluating through test_data', MLP_CLF_AC_1)
print('Accurracy:', MLP_CLF_AC_1 * 100, '%')

#confusion matrix
c_m = confusion_matrix(data_test_y, y_pred)
print(c_m)
#ploting
sns.heatmap(c_m, center=True)

# In[8]:


#let just give several different hidden layer size and alphas
H_l = [50,150,250]
A = [0.001, 0.0001]


# In[9]:


#this function choose best score through cross validation data
#based on given hidden layer size and alphas
def High_Score_Function_NT(hidden, a_v, X_tr, y_tr, X_cv, y_cv):      
    _sc = 0
    _hidden =[0] #empty
    _a_v = [0] #empty
    for i in hidden:
        for j in a_v :
            classifier = NT(hidden_layer_sizes=i, alpha=j)
            classifier.fit(X_tr, y_tr) # findig coefficients 
            sc = classifier.score(X_cv, y_cv) #tuning paramaters
            if sc > _sc: #choose high score
                _hidden = i #best hidden layer size 
                _a_v = j #best alpha
                _sc = sc  #best score
            print('Hidden_layer_size =', i,'alpha =', j , 'Score =', sc)
    return _hidden, _a_v


# In[10]:


h_l, a_v = High_Score_Function_NT(H_l, A, data_train_X , data_train_y, data_cv_X, data_cv_y) 


# In[11]:


MLP_CLF = NT(hidden_layer_sizes = h_l, alpha = a_v) #best alpha and best hidden layer size
MLP_CLF.fit(data_train_X, data_train_y) #finding coefficients 


# In[12]:


#finding accurrracy percentage 
MLP_CLF_AC_2 = round(MLP_CLF.score(data_test_X, data_test_y),4)
print('Evaluating through test_data', MLP_CLF_AC_2)
print('Accurracy:', MLP_CLF_AC_2 * 100, '%')


# # Support Vector Machines (SVM) + 

# In[13]:


from sklearn.svm import SVC


# In[14]:


SVC_CLF = SVC(kernel = 'linear', random_state = 0)
SVC_CLF.fit(data_train_X, data_train_y)
# Predicting the Test set results
y_pred = SVC_CLF.predict(data_test_X)
#finding accurracy_score
SVC_AC_1 = round(accuracy_score(data_test_y, y_pred),4)
print('Evaluating through test_data:', SVC_AC_1)
print('Accurracy:', SVC_AC_1 * 100, '%')


# In[15]:


from sklearn.svm import LinearSVC as SV
#let just give several different penalty paramater C  
C = [0.001, 0.01, 0.1, 1]


# In[16]:


#this function choose best score through cross validation data
#based on given penalty paramaeter C
def High_Score_Function_SVC(C, X_tr, y_tr, X_cv, y_cv):
     _sc = 0
     _C =[0]
     for i in C:
        classifier = SV(C=i)
        classifier.fit(X_tr, y_tr)
        sc = classifier.score(X_cv, y_cv) #tuning cross validation paramaters
        if sc > _sc:
            _C = i #finding best penalty parameter
            _sc = sc
        print('C =', i,'Score =', sc)
     return _C


# In[17]:


_c = High_Score_Function_SVC(C, data_train_X , data_train_y, data_cv_X, data_cv_y)     


# In[18]:


SVC_CLF = SV(C =_c)
SVC_CLF.fit(data_train_X, data_train_y) 
SVC_AC_2 = round(SVC_CLF.score(data_test_X, data_test_y),4)
print('Evaluating through test_data:', SVC_AC_2)
print('Accurracy:', SVC_AC_2 * 100, '%')


# # DECISION TREE +

# In[19]:


from sklearn.tree import DecisionTreeClassifier as DT


# In[20]:


# Fitting Decision Tree Classification to the Training set
DT_CLF = DT(criterion = 'entropy', random_state = 0)
DT_CLF.fit(data_train_X, data_train_y)
# Predicting the Test set results
y_pred = DT_CLF.predict(data_test_X)
#finding accurracy_score
DT_AC_1 = round(accuracy_score(data_test_y, y_pred),4)
print('Evaluating through test_data:', DT_AC_1)
print('Accurracy:', DT_AC_1 * 100, '%')


# In[21]:


#let just give several different max_depth and min_samples_split  
max_d = [16, 28]
min_s = [6, 12]


# In[22]:


#this function choose best score through cross validation data
#based on given different max_depth and min_samples_split 
def High_Score_Function_DT(max_d, min_s,  X_tr, y_tr, X_cv, y_cv):
    _sc = 0
    for i in max_d:
        for j in min_s:
            classifier = DT(max_depth=i, min_samples_split=j)
            classifier.fit(X_tr, y_tr)
            sc = classifier.score(X_cv, y_cv)
            if sc > _sc:
                _d = i 
                _s = j
                _sc = sc
                print('max_depth =',i,'min_samples =',j,'Score =',sc)
    return _d, _s


# In[23]:


_d, _s = High_Score_Function_DT(max_d, min_s, data_train_X , data_train_y, data_cv_X, data_cv_y)


# In[24]:


DT_CLF = DT(max_depth = _d, min_samples_split = _s)
DT_CLF.fit(data_train_X, data_train_y) 
DT_AC_2 = round(DT_CLF.score(data_test_X, data_test_y),4)
print('Evaluating through test_data:', DT_AC_2)
print('Accurracy:', DT_AC_2 * 100, '%')


# # EXTRA +

# In[25]:


#First step merging all data
data_merged = data_train.append(data_cv, ignore_index = True).append(data_test, ignore_index = True)


# In[26]:


#after merging data now this function will devide data into 3 parts
#60% for train, 20 % for test, and 20% for Cross validation
#then finding accuracy

def Average_Function(data, classifier):
    m = len(data)
    data_cv_sc = []
    data_test_sc = []
    
    for each in range(10):
        np.random.seed(each) #different random numbers into array
        index = np.arange(m)
        np.random.shuffle(index) #shuffle
        t = int(0.6 * m) # 60% data will be for training
        v = int(0.8 * m) # 20% for cross and 20 % for test
        
        train = index[:t]#60%
        cv = index[t:v] #20% 
        test = index[v:] #20%
        
        #loading data 
        data_train = data.loc[train]
        data_cv = data.loc[cv]
        data_test = data.loc[test]
        
        #features and values
        data_train_X = List_Function(data_train.iloc[:,1])
        data_test_X = List_Function(data_test.iloc[:,1])
        data_cv_X = List_Function(data_cv.iloc[:,1])
        
        #labels
        data_train_y = Values_Function(data_train.iloc[:,0])
        data_test_y = Values_Function(data_test.iloc[:,0])
        data_cv_y = Values_Function(data_cv.iloc[:,0])
        
        #finding accuracy for evaluating average performance
        classifier.fit(data_train_X, data_train_y)
        data_cv_sc.append(classifier.score(data_cv_X, data_cv_y))
        data_test_sc.append(classifier.score(data_test_X, data_test_y))
    
    return sum(data_cv_sc)/10, sum(data_test_sc)/10


# ### FOR NEURAL NETWORKS + 

# In[27]:


average_cv_sc_MLP, average_test_sc_MLP = Average_Function(data_merged, MLP_CLF)
print('Cross Validation score:' , average_cv_sc_MLP, ' Test score:', average_test_sc_MLP )
print('Cross Validation accuracy:' , round(average_cv_sc_MLP * 100,4),'%', ' Test accuracy:', round(average_test_sc_MLP * 100,4),'%' )


# ### FOR SVM +

# In[28]:


average_cv_sc_SVC, average_test_sc_SVC = Average_Function(data_merged, SVC_CLF)
print('Cross Validation score:' , average_cv_sc_SVC, ' Test score:', average_test_sc_SVC )
print('Cross Validation accuracy:' , round(average_cv_sc_SVC * 100,4),'%', ' Test accuracy:', round(average_test_sc_SVC * 100,4),'%' )


# ### FOR DECISION TREE +  

# In[29]:


average_cv_sc_DT, average_test_sc_DT = Average_Function(data_merged, DT_CLF)
print('Cross Validation score:' , average_cv_sc_DT, ' Test score:', average_test_sc_DT )
print('Cross Validation accuracy:' , round(average_cv_sc_DT * 100,4),'%', ' Test acurracy:', round(average_test_sc_DT * 100,4),'%' )

