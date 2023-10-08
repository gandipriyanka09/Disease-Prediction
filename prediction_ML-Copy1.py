#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,precision_score,roc_curve
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# **Read and shuffle the dataset**

# In[3]:


df = pd.read_csv(r"C:\Users\PRIYANKA\OneDrive\Desktop\Hack\dataset.csv")
df = shuffle(df,random_state=42)
df


# **Removing Hyphen from strings**

# In[4]:


df.info()


# In[5]:


for col in df.columns:
    
    df[col] = df[col].str.replace('_',' ')
df.head()


# **Dataset characteristics**

# In[6]:


df.describe()


# **Check for null and NaN values**

# In[7]:


null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_checker)


# In[ ]:





# **Remove the trailing space from the symptom columns**

# In[8]:


cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)
df.head()


# **Fill the NaN values with zero**

# In[9]:


df = df.fillna(0)
df.head()


# **Symptom severity rank**

# In[10]:


df1 = pd.read_csv(r"C:\Users\PRIYANKA\OneDrive\Desktop\Hack\Symptom-severity.csv")
df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
df1.head()


# **Get overall list of symptoms**

# In[11]:


df1['Symptom'].unique()


# **Encode symptoms in the data with the symptom rank**

# In[12]:


vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)
d.head()


# **Assign symptoms with no rank to zero**

# In[13]:


d = d.replace('dischromic  patches', 0)
d = d.replace('spotting  urination',0)
df = d.replace('foul smell of urine',0)
df.head(10)


# **Check if entire columns have zero values so we can drop those values**

# In[14]:


null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_checker)


# In[ ]:





# In[15]:


print("Number of symptoms used to identify the disease ",len(df1['Symptom'].unique()))
print("Number of diseases that can be identified ",len(df['Disease'].unique()))


# **Get the names of diseases from data**

# In[16]:


df['Disease'].unique()


# ### Select the features as symptoms column and label as Disease column
# 
# Explination: A **feature** is an input; **label** is an output.
# A feature is one column of the data in your input set. For instance, if you're trying to predict the type of pet someone will choose, your input features might include age, home region, family income, etc. The label is the final choice, such as dog, fish, iguana, rock, etc.
# 
# Once you've trained your model, you will give it sets of new input containing those features; it will return the predicted "label" (pet type) for that person.

# In[17]:


data = df.iloc[:,1:].values
labels = df['Disease'].values


# ## Splitting the dataset to training (80%) and testing (20%)
# 
# Separating data into training and testing sets is an important part of evaluating data mining models. Typically, when you separate a data set into a training set and testing set, most of the data is used for training, and a smaller portion of the data is used for testing. By using similar data for training and testing, you can minimize the effects of data discrepancies and better understand the characteristics of the model.
# After a model has been processed by using the training set, we test the model by making predictions against the test set. Because the data in the testing set already contains known values for the attribute that you want to predict, it is easy to determine whether the model's guesses are correct.
# 
# * Train Dataset: Used to fit the machine learning model.
# * Test Dataset: Used to evaluate the fit machine learning model.

# In[18]:


x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# ### Compute the F1 score, also known as balanced F-score or F-measure.
# 
# The F1 score can be interpreted as a weighted average of the precision and
# recall, where an F1 score reaches its best value at 1 and worst score at 0.
# The relative contribution of precision and recall to the F1 score are
# equal. The formula for the F1 score is
# 
#     F1 = 2 * (precision * recall) / (precision + recall)

# # Decision Tree

# In[19]:


tree =DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=13)
tree.fit(x_train, y_train)
preds=tree.predict(x_test)
conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)


# In[20]:


kfold = KFold(n_splits=10,shuffle=True,random_state=42)
DS_train =cross_val_score(tree, x_train, y_train, cv=kfold, scoring='accuracy')
pd.DataFrame(DS_train,columns=['Scores'])
print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_train.mean()*100.0, DS_train.std()*100.0))


# In[21]:


kfold = KFold(n_splits=10,shuffle=True,random_state=42)
DS_test =cross_val_score(tree, x_test, y_test, cv=kfold, scoring='accuracy')
pd.DataFrame(DS_test,columns=['Scores'])
print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_test.mean()*100.0, DS_test.std()*100.0))


# # Random Forest

# In[22]:


rfc=RandomForestClassifier(random_state=42)


# In[23]:


rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=13)
rnd_forest.fit(x_train,y_train)
preds=rnd_forest.predict(x_test)
print(x_test[0])
print(preds[0])
conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)


# In[24]:


kfold = KFold(n_splits=10,shuffle=True,random_state=42)
rnd_forest_train =cross_val_score(rnd_forest, x_train, y_train, cv=kfold, scoring='accuracy')
pd.DataFrame(rnd_forest_train,columns=['Scores'])
print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (rnd_forest_train.mean()*100.0, rnd_forest_train.std()*100.0))


# In[25]:


kfold = KFold(n_splits=10,shuffle=True,random_state=42)
rnd_forest_test =cross_val_score(rnd_forest, x_test, y_test, cv=kfold, scoring='accuracy')
pd.DataFrame(rnd_forest_test,columns=['Scores'])
print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (rnd_forest_test.mean()*100.0, rnd_forest_test.std()*100.0))


# # Fucntion to manually test the models

# In[26]:


discrp = pd.read_csv(r"C:\Users\PRIYANKA\OneDrive\Desktop\Hack\symptom_Description.csv")


# In[27]:


discrp.head()


# In[28]:


ektra7at = pd.read_csv(r"C:\Users\PRIYANKA\OneDrive\Desktop\Hack\symptom_precaution.csv")


# In[29]:


ektra7at.head()


# In[30]:


def predd(x,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17):
    psymptoms = [S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17]
    #print(psymptoms)
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]
    psy = [psymptoms]
    pred2 = x.predict(psy)
    disp= discrp[discrp['Disease']==pred2[0]]
    disp = disp.values[0][1]
    recomnd = ektra7at[ektra7at['Disease']==pred2[0]]
    c=np.where(ektra7at['Disease']==pred2[0])[0][0]
    precuation_list=[]
    for i in range(1,len(ektra7at.iloc[c])):
          precuation_list.append(ektra7at.iloc[c,i])
    print("The Disease Name: ",pred2[0])
    print("The Disease Discription: ",disp)
    print("Recommended Things to do at home: ")
    for i in precuation_list:
        print(i)


# **Test it Like The user would do**

# In[31]:


sympList=df1["Symptom"].to_list()
predd(rnd_forest,sympList[7],sympList[5],sympList[2],sympList[80],0,0,0,0,0,0,0,0,0,0,0,0,0)


# In[32]:


sympList=df1["Symptom"].to_list()
predd(rnd_forest,sympList[7],sympList[5],sympList[2],sympList[80],0,0,0,0,0,0,0,0,0,0,0,0,0)


# In[33]:


df1.iloc[2:9,:]


# In[34]:


df1.iloc[80:81,:2] #[rows inclusive:exclusive, cols inclusive:exclusive]


# In[35]:


predd(rnd_forest,"joint pain","acidity","shivering",0,0,0,0,0,0,0,0,0,0,0,0,0,0)


# In[36]:


predd(rnd_forest,"headache","high fever","vomiting","cough",0,0,0,0,0,0,0,0,0,0,0,0,0)


# In[ ]:





# In[ ]:




