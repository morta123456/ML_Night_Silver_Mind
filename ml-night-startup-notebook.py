#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as mse 
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import cross_val_score, KFold
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')


# # loading the data

# In[3]:


#pip install kaggle


# In[4]:


#!kaggle --version


# In[5]:


#!kaggle competitions download -c ml-night-gdsc-fst


# In[6]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('Submission Sample.csv')
train = train.set_index('campaign_id')
test = test.set_index('campaign_id')


# In[7]:


train.head()


# In[8]:


train.describe()


# ## Data Understanding

# In[9]:


train.info()


# In[10]:


train['start_date']=pd.to_datetime(train['start_date'])
train['end_date']=pd.to_datetime(train['end_date'])


# In[11]:


train['start_date:year'] = train['start_date'].dt.year
train['start_date:month'] = train['start_date'].dt.month
train['start_date:day'] = train['start_date'].dt.day


# In[12]:


train['end_date:year'] = train['end_date'].dt.year
train['end_date:month'] = train['end_date'].dt.month
train['end_date:day'] = train['end_date'].dt.day


# In[13]:


test['start_date']=pd.to_datetime(test['start_date'])
test['end_date']=pd.to_datetime(test['end_date'])


# In[14]:


test['start_date:year'] = test['start_date'].dt.year
test['start_date:month'] = test['start_date'].dt.month
test['start_date:day'] = test['start_date'].dt.day


# In[15]:


test['end_date:year'] = test['end_date'].dt.year
test['end_date:month'] = test['end_date'].dt.month
test['end_date:day'] = test['end_date'].dt.day


# ### Handeling Categorical Variables 

# In[16]:


train['chain_id'].value_counts()


# In[17]:


fig = plt.figure(figsize=(15,10))
sns.barplot(x=train['chain_id'],y=train['budget'],data=train)
plt.show()


# In[18]:


dummies = pd.get_dummies(train['chain_id'], prefix='cat',drop_first=True)
train =pd.concat([train, dummies], axis=1)
train.drop('chain_id',axis=1,inplace=True)
train.head()


# In[19]:


dummies = pd.get_dummies(test['chain_id'], prefix='cat',drop_first=True)
test =pd.concat([test, dummies], axis=1)
test.drop('chain_id',axis=1,inplace=True)
test.head()


# In[20]:


train['format'].value_counts()


# In[21]:


fig = plt.figure(figsize=(15,10))
sns.barplot(x=train['format'],y=train['budget'],data=train)
plt.show()


# In[22]:


dummies = pd.get_dummies(train['format'], prefix='Category',drop_first=True)
train =pd.concat([train, dummies], axis=1)
train.drop('format',axis=1,inplace=True)
train.drop('device',axis=1,inplace=True)
train


# In[23]:


dummies = pd.get_dummies(test['format'], prefix='Category',drop_first=True)
test =pd.concat([test, dummies], axis=1)
test.drop('format',axis=1,inplace=True)
test.drop('device',axis=1,inplace=True)
test.head()


# In[24]:


fig = plt.figure(figsize=(15,10))
sns.scatterplot(x='days',y='budget',data=train)
plt.show()


# In[25]:


train['start_day'].value_counts()


# In[26]:


train['end_day'].value_counts()


# In[27]:


train['shop'].value_counts()


# In[ ]:





# # data processing

# ### Smote

# In[28]:


col = train['shop']
# Convert the column to a numpy array
col = col.values.reshape(-1, 1)


# In[29]:


smote = SMOTE(sampling_strategy='minority')
#col_resampled, y_resampled = smote.fit_resample(col, train['budget'])


# In[30]:


##data_resampled = data.copy()
#data_resampled[column_name] = col_resampled.flatten()


# # Correlations

# In[31]:


# Visualizing the collelations between all variables of the data.
plt.figure(1 , figsize = (20,10))
cor = sns.heatmap(train.corr(), annot = True)
cor.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
plt.show()


# In[32]:


plt.figure(1 , figsize = (20,10))
train.corr()['budget'].sort_values(ascending = False).plot(kind='bar', color=['#FFAEBC','#A0E7E5', '#B4F8C8', '#FBE7C6', '#79BEEE'])
plt.show()


# In[33]:


#pd.plotting.scatter_matrix(train, alpha=0.5, figsize=(30, 22), diagonal='kde')
#plt.show()


# ### Feature Distributions

# In[34]:


train.head()


# In[35]:


train['height'].plot(kind='kde')
plt.show()


# In[36]:


#train['height'],_ = pd.Series(stats.boxcox(train['height']))
#test['height'],_ = pd.Series(stats.boxcox(test['height']))


# In[37]:


train['width'].plot(kind='kde')
plt.show()


# In[38]:


#train['width'],_ = pd.Series(stats.boxcox(train['width']))
#test['width'],_ = pd.Series(stats.boxcox(test['width']))


# In[39]:


train['budget'].plot(kind='kde')
plt.show()


# In[40]:


#train['budget'],_ = pd.Series(stats.boxcox(train['budget']))


# In[41]:


train['budget'].plot(kind='kde')
plt.show()


# # creating the X and y 

# In[42]:


y = train[['budget']]


# In[43]:


x = train.drop(['start_date','end_date','iremoteid','shop','budget'],axis=1)
test = test.drop(['start_date','end_date','iremoteid','shop'],axis=1)


# In[44]:


X_train , X_test , y_train ,y_test = train_test_split(x,y,test_size = 0.1,random_state =4)


# In[45]:


test.head()


# In[46]:


x.head()


# In[47]:


y.head()


# # creating the model

# In[48]:


model = XGBRegressor()
model.fit(X_train,y_train)


# ### Cross validation

# In[49]:


#KFold cross-validation with 10 folds 
kf = KFold(n_splits=10,shuffle=True,random_state=7)
cv_results1 =cross_val_score(model,x,y,cv=kf,scoring='neg_root_mean_squared_error')


# In[50]:


print(cv_results1)


# In[51]:


print(np.mean(cv_results1), np.std(cv_results1))#


# In[52]:


print(np.quantile(cv_results1,[0.025,0.975]))


# # evaluating the model 

# In[53]:


# note that the lower the MSE the better 
y_preds = model.predict(X_test)
rmse = mse(y_test,y_preds)**(1/2)
print(rmse)


# In[54]:


y_train.head()


# In[55]:


y_test.head()


# ### Features Importance

# In[56]:


print(model.feature_importances_)


# In[57]:


cmap = plt.cm.get_cmap('tab20c_r')
colors = [cmap(i) for i in range(x.shape[1])]
fig, ax = plt.subplots(figsize=(12, 7))
plot_importance(model, ax=ax, color=colors)
plt.title('Feature Importances', fontdict={'fontsize':18}, pad=12)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(True)
plt.show()


# x=x.drop('Category_highco',axis=1)
# test=test.drop('Category_highco',axis=1)
# X_train , X_test , y_train ,y_test = train_test_split(x,y,test_size = 0.1,random_state = 7)

# In[58]:


x.head()


# In[59]:


test.head()


# ### Grid Search

# In[60]:


xgb_params = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [100, 200, 300, 400,1000],
    'subsample': [0.5, 0.75, 1],
    'colsample_bytree': [0.5, 0.75, 1]
}


# In[61]:


grid_search = GridSearchCV(model, xgb_params, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)


# In[62]:


#grid_search.fit(X_train, y_train)


# In[63]:


print('Best parameters: ', grid_search.best_params_)


# In[ ]:


model = XGBRegressor(colsample_bytree=1,learning_rate=0.1, max_depth= 5, n_estimators=10000, subsample= 1,gamma=0,min_child_weight=1,reg_alpha=0,reg_lambda=1,scale_pos_weight=1)
model.fit(X_train,y_train)


# In[ ]:


# note that the lower the MSE the better 
y_preds = model.predict(X_test)
rmse = mse(y_test,y_preds)**(1/2)
print(rmse)


# # creating the submission

# In[ ]:


preds = model.predict(test)


# In[ ]:


submission['budget'] = preds


# In[ ]:


# this code will generate a file that you should download 
submission.to_csv('ss.csv',index=False)


# In[ ]:




