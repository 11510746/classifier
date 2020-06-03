#%%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import utils

# https://www.bbsmax.com/A/q4zVlo1l5K/
# prepare data
x,y = utils.load_data_as_df('dataset/train.data')
smote = SMOTE(sampling_strategy=0.5, random_state=100)
x, y = smote.fit_sample(x, y)

k = 5

best_params = {'random_state':10}
#%% n_estimators
parameters = {
    'n_estimators':range(1,101,10),  # number of trees, default = 100
}

clf = GridSearchCV(RandomForestClassifier(**best_params), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

results = search.cv_results_
print('best prarams', search.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'n_estimators': [ param_dic['n_estimators'] for param_dic in results['params'] ],
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='n_estimators', y='mean fit time', data=df, ax=axes[0], 
    markers=True )
sns.lineplot(x='n_estimators', y='mean f1 score', data=df, ax=axes[1], 
    markers=True )
fig.savefig('img/GS_GRDBST_estimators.png')
# best prarams {'n_estimators': 11}
# %% max_depth, min_samples_split
best_params['n_estimators'] = search.best_params_['n_estimators']
parameters = {
    'max_depth': range(3,14,2),
    'min_samples_split': range(30,200,30)
}

clf = GridSearchCV(RandomForestClassifier(**best_params), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

results = search.cv_results_
print('best prarams', search.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'max_depth': [ param_dic['max_depth'] for param_dic in results['params'] ],
        'min_samples_split': [ param_dic['min_samples_split'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='min_samples_split', y='mean fit time',style='max_depth', data=df, ax=axes[0], 
    markers=True )
sns.lineplot(x='min_samples_split', y='mean f1 score', style='max_depth',data=df, ax=axes[1], 
    markers=True )
fig.savefig('img/GS_GRDBST_depth_minsample.png')
# best prarams {'max_depth': 13, 'min_samples_split': 30}

# %%
best_params['max_depth'] = search.best_params_['max_depth']
parameters = {
    'min_samples_leaf': range(10,50,10),
    'min_samples_split': range(10,100,20)
}

clf = GridSearchCV(RandomForestClassifier(**best_params), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

results = search.cv_results_
print('best prarams', search.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'min_samples_leaf': [ param_dic['min_samples_leaf'] for param_dic in results['params'] ],
        'min_samples_split': [ param_dic['min_samples_split'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='min_samples_split', y='mean fit time',style='min_samples_leaf', data=df, ax=axes[0], 
    markers=True )
sns.lineplot(x='min_samples_split', y='mean f1 score', style='min_samples_leaf',data=df, ax=axes[1], 
    markers=True )
fig.savefig('img/GS_GRDBST_minleaf_minsplit.png')

# best prarams {'min_samples_leaf': 20, 'min_samples_split': 30}

# %% max_features
best_params['min_samples_leaf'] = search.best_params_['min_samples_leaf']
best_params['min_samples_split'] = search.best_params_['min_samples_split']
parameters = {'max_features': [0.2, 0.4,0.6,0.8,1]}

clf = GridSearchCV(RandomForestClassifier(**best_params), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

results = search.cv_results_
print('best prarams', search.best_params_)
# %%
df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'max_features': [ param_dic['max_features'] for param_dic in results['params'] ],
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='max_features', y='mean fit time', data=df, ax=axes[0], 
    markers=True )
sns.lineplot(x='max_features', y='mean f1 score',data=df, ax=axes[1], 
    markers=True )
fig.savefig('img/GS_GRDBST_minleaf_minsplit.png')
# best prarams {'max_features': 0.8}

# %%
