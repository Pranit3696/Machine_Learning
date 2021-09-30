#!/usr/bin/env python
# coding: utf-8

# # Name: Pranit Prabhakaran
# ## LGM CID -LGM VIPDSWL0001357
# 
# ### Topic :- Iris Flowers Classification ML Project 

#  # Importing Data

# In[241]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')


# In[242]:


cd C:\Users\prani\Desktop\DataAnalysis


# In[243]:


df = pd.read_csv("irisdata.csv",header = None)
df.head()


# # Data Observation

# In[222]:


df_head = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
df.to_csv("Iris.csv",header = df_head,index = False)
df1 = pd.read_csv("Iris.csv")
df1.head()

#For top 5 rows


# # Exploratory Data Analysis (EDA)

# In[223]:


# Number of rows and columns in iris data set


# In[244]:


print(df1.shape)


# In[245]:


print(df1.columns)


# In[246]:


print (df1.info)


# In[196]:


df1.describe()


# In[227]:


df1.isnull().sum()                                   # Checking null values in dataset


# In[228]:


df1.count()                                                     # Counting the datavalues of each column


# # Grouping
# 

# In[199]:


df1.groupby('Species').size()                                    # Checking for outliers


# In[229]:


fig,ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=df1,width = 0.5,ax=ax,fliersize = 3)


# In[201]:


fig,ax = plt.subplots(figsize = (5,5))
sns.boxplot(data = df1["SepalWidthCm"],width = 0.8,ax=ax, fliersize = 5)


# # Plotting 

# In[202]:


sns.pairplot(data = df1,hue = 'Species')
plt.show()


# # Correlation

# In[230]:


df1.corr()


# In[247]:


plt.subplots(figsize = (8,7))
sns.heatmap(df1.corr(),annot = True,fmt = "f").set_title("To show the correlation of attributes in Iris Species")


# From the above heat map we can draw observations that the Petal Length and Petal Width are highly correlated.

# #  Scatter Plot showing sepal length and sepal width

# In[248]:


setosa = df1[df1['Species'] == "Iris-setosa"]
versicolor = df1[df1['Species'] == "Iris-versicolor"]
virginica = df1[df1['Species'] == "Iris-virginica"]

plt.figure(figsize = (10,10))
plt.scatter(setosa['SepalLengthCm'],setosa['SepalWidthCm'],c = "yellow",label="Iris-setosa",marker = '<')
plt.scatter(versicolor['SepalLengthCm'],versicolor['SepalWidthCm'],c = "red",label="Versicolor",marker = '>')
plt.scatter(virginica['SepalLengthCm'],versicolor['SepalWidthCm'],c = "green",label="Virginica",marker = '*')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width",fontsize = 15)
plt.legend()
plt.show()


# # Scatter Plot showing petal lenth and petal width

# In[233]:


plt.scatter(setosa['PetalLengthCm'],setosa['PetalWidthCm'],c="blue",label="Iris-setosa",marker = '*')
plt.scatter(versicolor['PetalLengthCm'],versicolor['PetalWidthCm'],c ="red",label="Versicolor",marker ='^')
plt.scatter(virginica['PetalLengthCm'],virginica['PetalWidthCm'],c = "green",label="Virginica",marker = '<')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Petal Width",fontsize = 10)
plt.legend()
plt.show()


# # Spliting the dataset

# In[207]:


x = df1.drop(columns = 'Species')
y = df1['Species']


# In[208]:


# Visual description of data distributed for every column
plt.figure(figsize = (20,20),facecolor = 'white')
plotnumber = 1

for column in x:
    if plotnumber<=1:
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(y,x[column])
    plotnumber+=1
plt.tight_layout()


# In[ ]:





# #  Model Selection
# 

# In[209]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4 , random_state = 1)


# In[210]:


print("x_train:",len(x_train))
print("x_test:",len(x_test))
print("y_train:",len(y_train))
print("y_test:",len(y_test))


# # Building Models

# # 1.Logistic Regression

# In[234]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score


# In[235]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[236]:


predict = model.predict(x_test)
print('Predicted Values on test Data',predict)


# In[237]:


y_test_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)


# In[238]:


print("Training Accuracy :",accuracy_score(y_train,y_train_pred))
print("Test Accuracy :",accuracy_score(y_test,y_test_pred))


# # 2. Decision Tree

# In[239]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 20)
classifier.fit(x_train,y_train)


# In[240]:


y_pred = classifier.predict(x_test)


# In[218]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[219]:


from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# Conclusion :- Both the models give same accuracy of 96%

# In[ ]:




