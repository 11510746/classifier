#%%
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import utils

#%%
# http://sofasofa.io/forum_main_post.php?postid=1000868

k=5
# prepare data
x,y = utils.load_data_as_df('dataset/train.data')
smote = SMOTE(sampling_strategy=0.5, random_state=100)
x, y = smote.fit_sample(x, y)

best_params = {}
#%% max_depth, min_child_weight
parameters = {
    'max_depth':[3,5,7,9],
    'min_child_weight':[1,3,5]
    }
gsearch = GridSearchCV(estimator=XGBClassifier(), param_grid=parameters, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'max_depth': [ param_dic['max_depth'] for param_dic in results['params'] ],
        'min_child_weight': [ param_dic['min_child_weight'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='max_depth', y='mean fit time', hue='min_child_weight', data=df, ax=axes[0], 
    markers=True, style='min_child_weight')
sns.lineplot(x='max_depth', y='mean f1 score', hue='min_child_weight', data=df, ax=axes[1], 
    markers=True, style='min_child_weight')
fig.savefig('img/GS_XGB_maxdepth_minchildweight.png')

# best prarams {'max_depth': 9, 'min_child_weight': 1}

# %% learning_rate, n_estimators
best_params['max_depth'] = gsearch.best_params_['max_depth']
best_params['min_child_weight'] = gsearch.best_params_['min_child_weight']

parameters = {
    'learning_rate':[0.001,0.01,0.1],
    'n_estimators': range(200,301,20)
}
gsearch = GridSearchCV(estimator=XGBClassifier(**best_params), param_grid=parameters, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'learning_rate': [ param_dic['learning_rate'] for param_dic in results['params'] ],
        'n_estimators': [ param_dic['n_estimators'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='n_estimators', y='mean fit time', hue='learning_rate', data=df, ax=axes[0], 
    markers=True, style='learning_rate')
sns.lineplot(x='n_estimators', y='mean f1 score', hue='learning_rate', data=df, ax=axes[1], 
    markers=True, style='learning_rate')
fig.savefig('img/GS_XGB_estimators_learnrate.png')

# best prarams {'learning_rate': 0.1, 'n_estimators': 220}

#%% colsample_bytree,subsample
best_params['learning_rate'] = gsearch.best_params_['learning_rate']
best_params['n_estimators'] = gsearch.best_params_['n_estimators']

parameters = {
    'colsample_bytree':[0.5,0.6,0.8,1],
    'subsample': [0.5,0.6,0.8,1]
}
gsearch = GridSearchCV(estimator=XGBClassifier(**best_params), param_grid=parameters, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'colsample_bytree': [ param_dic['colsample_bytree'] for param_dic in results['params'] ],
        'subsample': [ param_dic['subsample'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='colsample_bytree', y='mean fit time', hue='subsample', data=df, ax=axes[0], 
    markers=True, style='subsample')
sns.lineplot(x='colsample_bytree', y='mean f1 score', hue='subsample', data=df, ax=axes[1], 
    markers=True, style='subsample')
fig.savefig('img/GS_XGB_col_subsample.png')

# best prarams {'colsample_bytree': 1, 'subsample': 0.6}

#%%
best_params['colsample_bytree'] = gsearch.best_params_['colsample_bytree']
best_params['subsample'] = gsearch.best_params_['subsample']

parameters = {
    'reg_alpha':[0.01, 0.1, 0.5, 0.8, 1],
    'reg_lambda': [0.01, 0.1, 0.5, 0.8, 1]
}
gsearch = GridSearchCV(estimator=XGBClassifier(**best_params), param_grid=parameters, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)
#%%
df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'reg_alpha': [ param_dic['reg_alpha'] for param_dic in results['params'] ],
        'reg_lambda': [ param_dic['reg_lambda'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='reg_alpha', y='mean fit time', hue='reg_lambda', data=df, ax=axes[0], 
    markers=True, style='reg_lambda')
sns.lineplot(x='reg_alpha', y='mean f1 score', hue='reg_lambda', data=df, ax=axes[1], 
    markers=True, style='reg_lambda')
fig.savefig('img/GS_XGB_alpha_lambda.png')

# best prarams {'reg_alpha': 1, 'reg_lambda': 1}