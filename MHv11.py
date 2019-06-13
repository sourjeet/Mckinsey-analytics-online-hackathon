
# coding: utf-8

# In[1]:

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score, GridSearchCV #score evaluation
from sklearn.model_selection import cross_val_predict #prediction


# In[2]:


from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE 

from collections import Counter

import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[3]:

train = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")


# In[4]:

train = train.set_index('id')
test_pid=test.pop('id')


# In[5]:

train_stage = train.copy()
test_stage = test.copy()


# In[6]:

test_stage.head(10)


# In[7]:

combine = [train_stage, test_stage]


# In[23]:

train_stage.describe()


# In[24]:

test_stage.describe()


# In[10]:

train_stage['age_Range']=pd.qcut(train_stage['age'],13)
train_stage.groupby(['age_Range'])['stroke'].mean().to_frame().style.background_gradient(cmap='summer_r')


# In[11]:

sns.distplot(train_stage.avg_glucose_level)


# In[12]:

train_stage['AGL_Range']=pd.qcut(train_stage.loc[(train_stage['avg_glucose_level']>140),'avg_glucose_level'],8)
train_stage.groupby(['AGL_Range'])['stroke'].mean().to_frame().style.background_gradient(cmap='summer_r')


# In[13]:

for dataset in combine:
    dataset['age_cat']=0
    #train_stage.loc[train_stage['GrossIncome']<=3650,'GrossIncome_cat']=0
    dataset.loc[(dataset['age']>11)&(dataset['age']<=22),'age_cat']=1
    dataset.loc[(dataset['age']>22)&(dataset['age']<=31),'age_cat']=2
    dataset.loc[(dataset['age']>31)&(dataset['age']<=40),'age_cat']=3
    dataset.loc[(dataset['age']>40)&(dataset['age']<=47),'age_cat']=4
    dataset.loc[(dataset['age']>47)&(dataset['age']<=54),'age_cat']=5
    dataset.loc[(dataset['age']>54)&(dataset['age']<=62),'age_cat']=6
    dataset.loc[(dataset['age']>62)&(dataset['age']<=72),'age_cat']=7
    dataset.loc[(dataset['age']>72),'age_cat']=9
    
    dataset['AGL_cat']=0
    dataset.loc[(dataset['avg_glucose_level']>140)&(dataset['avg_glucose_level']<=170),'AGL_cat']=1
    dataset.loc[(dataset['avg_glucose_level']>170),'AGL_cat']=2    

train_stage = train_stage.drop(['age_Range','age'], axis=1)
test_stage = test_stage.drop(['age'], axis=1)

train_stage = train_stage.drop(['AGL_Range','avg_glucose_level'], axis=1)
test_stage = test_stage.drop(['avg_glucose_level'], axis=1)


# In[14]:

train_stage.pivot_table(index = ['age_cat','AGL_cat'], values = 'stroke', aggfunc=np.mean)


# In[15]:

train_stage['smoking_status'].fillna('never smoked',inplace = True)
test_stage['smoking_status'].fillna('never smoked',inplace = True)


# In[16]:

cat_var=train_stage.select_dtypes(include = ['object']).columns
for var in cat_var :
    if(var not in ['stroke']):
        print(var)
        print(train_stage[var].unique())
        print(test_stage[var].unique())


# In[17]:

train_stage.pivot_table(index = ['work_type'], values = 'stroke', aggfunc=np.mean)


# In[18]:

train_stage.pivot_table(index = ['smoking_status'], values = 'stroke', aggfunc=np.mean)


# In[19]:

train_stage['gender'].replace(['Male','Female', 'Other'],[2,1,0],inplace=True)
train_stage['ever_married'].replace(['No','Yes'],[0,1],inplace=True)
train_stage['work_type'].replace(['children','Never_worked','Self-employed','Private', 'Govt_job'],[0,0,2,1,1],inplace=True)
train_stage['Residence_type'].replace(['Rural','Urban'],[0,1],inplace=True)
train_stage['smoking_status'].replace(['never smoked','formerly smoked','smokes'],[0,2,1],inplace=True)

test_stage['gender'].replace(['Male','Female', 'Other'],[2,1,0],inplace=True)
test_stage['ever_married'].replace(['No','Yes'],[0,1],inplace=True)
test_stage['work_type'].replace(['children','Never_worked','Self-employed','Private', 'Govt_job'],[0,0,2,1,1],inplace=True)
test_stage['Residence_type'].replace(['Rural','Urban'],[0,1],inplace=True)
test_stage['smoking_status'].replace(['never smoked','formerly smoked','smokes'],[0,2,1],inplace=True)


# In[20]:

table = train_stage.pivot_table(index = ['work_type','age_cat','hypertension','AGL_cat'], values = 'bmi', aggfunc=np.mean)
print(table)


# In[21]:

def fill(x):
    if pd.isnull(x['bmi']):
        return table.loc[x['work_type'],x['age_cat'],x['hypertension'],x['AGL_cat']].values
    else:
        return x['bmi']


# In[22]:

train_stage['bmi'] = train_stage.apply(lambda x : round(float(fill(x)),2),axis=1)
test_stage['bmi'] = test_stage.apply(lambda x : round(float(fill(x)),2),axis=1)


# In[25]:

for column in ['bmi']:
    print(column)
    #np.log(train_stage[column]).hist(bins = 100)
    np.log(train_stage[column]).hist(bins=100)
    plt.show()


# In[26]:

train_stage['log_bmi'] = np.log(train_stage['bmi'])
test_stage['log_bmi'] = np.log(test_stage['bmi'])

train_stage = train_stage.drop(['bmi'], axis=1)
test_stage = test_stage.drop(['bmi'], axis=1)


# In[ ]:

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.distplot(train_stage[train_stage['stroke']==0].bmi,ax=ax[0])
ax[0].set_title('no stroke')
sns.distplot(train_stage[train_stage['stroke']==1].bmi,ax=ax[1])
ax[1].set_title('stroke')
plt.show()


# In[ ]:

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.distplot(train_stage[train_stage['stroke']==0].avg_glucose_level,ax=ax[0])
ax[0].set_title('no stroke')
sns.distplot(train_stage[train_stage['stroke']==1].avg_glucose_level,ax=ax[1])
ax[1].set_title('stroke')
plt.show()


# In[27]:

sns.heatmap(train_stage.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[41]:

train_stage = train_stage.drop(['Residence_type'], axis=1)
test_stage = test_stage.drop(['Residence_type'], axis=1)


# In[42]:

y=train_stage['stroke']
X = train_stage.drop('stroke', axis=1)


# In[43]:

sm = SMOTE(random_state=21)


# In[44]:

X_resampled, y_resampled = sm.fit_sample(X, y)


# In[45]:

X_resampled = pd.DataFrame(X_resampled)


# In[46]:

X_resampled.columns = X.columns


# In[47]:

print(X.shape)
print(y.shape)
print(X_resampled.shape)
print(y_resampled.shape)


# In[48]:

np.sum(y)


# In[49]:

X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=42, stratify=y_resampled)


# In[50]:

model_rfc= RandomForestClassifier()


# In[51]:

model_rfc.fit(X_train, y_train)


# In[52]:

predicted= model_rfc.predict_proba(X_test)


# In[53]:

fpr, tpr, thresholds = roc_curve(y_test, predicted[:,1], pos_label=1)
auc_algo = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_algo)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[54]:

feature_importance = pd.Series(model_rfc.feature_importances_, index=X.columns)
columns = feature_importance[feature_importance > 0.0025].index
print(columns)
feature_importance.sort_values(inplace=True)
feature_importance.plot(kind='barh',figsize=(17,20));
plt.show()


# In[ ]:




# In[ ]:




# In[55]:

xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=6,
 min_child_weight=6,
 gamma=0.09,
 reg_alpha = 1,
 subsample=0.8,
 colsample_bytree=0.7,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


# In[56]:

xgb2.fit(X_train, y_train)


# In[57]:

y_predict = xgb2.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_predict[:,1], pos_label=1)
auc_algo = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_algo)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[58]:

xgb2.fit(X_resampled, y_resampled)

_pred = xgb2.predict_proba(test_stage)

pred_reSample_xgb2 = pd.DataFrame(_pred[:,1],columns=['stroke'])

Prediction = pd.concat([test_pid,pred_reSample_xgb2], axis=1 )
Prediction.to_csv('../Submissions/Submission11.csv',sep=',',index =False)


# In[ ]:



