#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ARDRegression, TweedieRegressor, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


# In[3]:


plt.rcParams['figure.figsize'] = (12,8)
pd.set_option('display.float_format',lambda x: '%.3f' % x)


# In[4]:


df = pd.read_csv('health_insurance.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df['bmi'].fillna(df['bmi'].mean(),inplace=True)
df['age'].fillna(df['age'].mode()[0],inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df[df.duplicated()]


# In[11]:


df['hereditary_diseases'].unique()


# In[12]:


df['city'].nunique()


# In[13]:


df['job_title'].value_counts().head()


# In[14]:


values = df.sex.value_counts().values
labels = ['female','male']
explode = (0.2,0)

plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.2f%%')
plt.title('Gender',pad=32,fontsize=26,fontweight='bold',color='fuchsia')
plt.show()


# In[16]:


values = df.smoker.value_counts().values
labels = df.smoker.value_counts().keys()
explode = (0.2,0)

plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.2f%%',colors=['red','green'])
plt.title('Smoker')
plt.show()


# In[24]:


values = df.diabetes.value_counts().values
labels = df.diabetes.value_counts().keys()
explode = (0.2,0)

plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.2f%%',colors=['green','orchid'])
plt.title('Diabetes')
plt.show()


# In[18]:


values = df.regular_ex.value_counts().values
labels = df.regular_ex.value_counts().keys()
explode = (0.2,0)

plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.2f%%',colors=['orangered','lawngreen'])
plt.title('Regular Exercise')
plt.show()


# In[29]:


sns.pairplot(df,hue='sex')


# In[30]:


sns.heatmap(df.corr(),annot=True,cmap='viridis',vmin=-1,vmax=1)


# In[ ]:


job_titles_bmi = df.groupby('job_title')['bmi'].mean().sort_values(ascending=False).reset_index()[:10]
job_titles_bmi


# In[ ]:


sns.barplot(x='bmi',y='job_title',data=job_titles_bmi,palette='rainbow',orient='horizontal')


# In[ ]:


job_titles_bmi = df.groupby('job_title')['bmi'].max().sort_values(ascending=False).reset_index()[:10]
job_titles_bmi


# In[ ]:


sns.barplot(x='job_title',y='bmi',data=job_titles_bmi,palette='crest')
plt.xticks(rotation=20)
plt.title('Job professionals with highest BMI')


# In[ ]:


job_titles_bmi = job_titles_bmi = df.groupby('job_title')['bmi'].min().sort_values().reset_index()[:10]
job_titles_bmi


# In[ ]:


sns.barplot(x='job_title',y='bmi',data=job_titles_bmi,palette='winter')


# In[ ]:


plt.figure(figsize=(20,10))
pd.crosstab(index=df.job_title,columns=df.sex,values=df.claim,normalize='index',aggfunc='mean').plot.bar(stacked=True,color=['crimson','royalblue'])
plt.ylabel('claim')
plt.legend(bbox_to_anchor=(1.2,0.5),title='sex')


# In[ ]:


sns.countplot(df.hereditary_diseases)


# In[ ]:


plt.figure(figsize=(20,10))
pd.crosstab(index=df.age,columns=df.smoker,values=df.claim,aggfunc='mean',normalize='index').plot.bar(stacked=True,color=['red','green'])
plt.ylabel('claim')
plt.legend(bbox_to_anchor=(1.2,0.5),title='smoker')


# In[ ]:


sns.boxplot(df.claim)


# In[ ]:


ax = sns.barplot(x='sex',y='claim',data=df)


# In[ ]:


sns.barplot(x='smoker',y='claim',data=df)


# In[ ]:


sns.stripplot(x='regular_ex',y='claim',data=df,hue='sex')


# In[ ]:


sns.violinplot(x='diabetes',y='claim',data=df,hue='sex')


# In[ ]:


df.sex.replace(['female','male'],[0,1],inplace=True)
df.sex = df.sex.astype(int)


# In[ ]:


df.info()


# In[ ]:


df.sex.replace(['female','male'],[0,1],inplace=True)
df.sex = df.sex.astype(int)


# In[ ]:


df.info()


# In[ ]:


le = LabelEncoder()
df.city = le.fit_transform(df.city)
df.city = df.city.astype(int)
df.job_title = le.fit_transform(df.job_title)
df.job_title = df.job_title.astype(int)
df.hereditary_diseases = le.fit_transform(df.hereditary_diseases)
df.hereditary_diseases = df.hereditary_diseases.astype(int)


# In[ ]:


scaler = StandardScaler()
features = df.columns
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df,columns=features)
scaled_df.head()


# In[ ]:




