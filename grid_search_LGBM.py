#%%
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import lightgbm as lgb

import utils
#%%

k=5
# prepare data
x,y = utils.load_data_as_df('dataset/train.data')
smote = SMOTE(sampling_strategy=0.5, random_state=100)
x, y = smote.fit_sample(x, y)

#%% max_depth, num_leaves
parameters1 = {
    'max_depth': [4,6,8], # 太深容易过拟合
    'num_leaves': [20,30,40]
    }
gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(objective = 'binary'), param_grid=parameters1, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)
#%%
df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'max_depth': [ param_dic['max_depth'] for param_dic in results['params'] ],
        'num_leaves': [ param_dic['num_leaves'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='num_leaves', y='mean fit time', hue='max_depth', data=df, ax=axes[0], 
    markers=True, style='max_depth')
sns.lineplot(x='num_leaves', y='mean f1 score', hue='max_depth', data=df, ax=axes[1], 
    markers=True, style='max_depth')
fig.savefig('img/GS_LGBM_maxdepth_numleaves.png')


# best prarams {'max_depth': 8, 'num_leaves': 40}

# %% min_data_in_leaf,min_sum_hessian_in_leaf,防止树过拟合
parameters2 = {
    'min_child_samples': [18,19,20,21,22], 
    'min_child_weight': [0.001,0.002]
}
gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(objective = 'binary',max_depth=8,num_leaves=40), param_grid=parameters2, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'min_child_samples': [ param_dic['min_child_samples'] for param_dic in results['params'] ],
        'min_child_weight': [ param_dic['min_child_weight'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='min_child_samples', y='mean fit time', hue='min_child_weight', data=df, ax=axes[0], 
    markers=True, style='min_child_weight')
sns.lineplot(x='min_child_samples', y='mean f1 score', hue='min_child_weight', data=df, ax=axes[1], 
    markers=True, style='min_child_weight')
fig.savefig('img/GS_GRDBST_minchildsamples_weight.png')
# best prarams {'min_child_samples': 18, 'min_child_weight': 0.001}

# %% 调整feature_fraction,防止过拟合
parameters3 = {'feature_fraction': [0.6, 0.8, 1]}
gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(objective = 'binary',max_depth=8,
                                        num_leaves=40,min_child_weight=0.001,min_child_samples=18), 
                        param_grid=parameters3, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'feature_fraction': [ param_dic['feature_fraction'] for param_dic in results['params'] ]
    }
)

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='feature_fraction', y='mean fit time',data=df,ax=axes[0])
sns.lineplot(x='feature_fraction', y='mean f1 score',data=df,ax=axes[1])
fig.savefig('img/GS_GRDBST_featurefraction.png')
# best prarams {'feature_fraction': 0.8}

# %% 调整bagging_fraction和bagging_freq
parameters4 = {
    'bagging_fraction': [0.8,0.9,1],
    'bagging_freq': [2,3,4],
}
gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(objective = 'binary',max_depth=8,
                                        num_leaves=40,min_child_weight=0.001,min_child_samples=18,
                                        feature_fraction=0.8), 
                                    param_grid=parameters4, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'bagging_fraction': [ param_dic['bagging_fraction'] for param_dic in results['params'] ],
        'bagging_freq': [ param_dic['bagging_freq'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='bagging_fraction', y='mean fit time', hue='bagging_freq', data=df, ax=axes[0], 
    markers=True, style='bagging_freq')
sns.lineplot(x='bagging_fraction', y='mean f1 score', hue='bagging_freq', data=df, ax=axes[1], 
    markers=True, style='bagging_freq')
fig.savefig('img/GS_GRDBST_bagging_freq_fraction.png')
# best prarams {'bagging_fraction': 0.9, 'bagging_freq': 3}

# %% cat_smooth为设置每个类别拥有最小的个数，主要用于去噪
parameters4 = {'cat_smooth': [0,10,20]}
gsearch = GridSearchCV(estimator=lgb.LGBMClassifier(objective = 'binary',max_depth=8,
                                        num_leaves=40,min_child_weight=0.001,min_child_samples=18,
                                        feature_fraction=0.8,bagging_fraction=0.9, bagging_freq=3), 
                                    param_grid=parameters4, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'cat_smooth': [ param_dic['cat_smooth'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='cat_smooth', y='mean fit time', data=df, ax=axes[0], 
    markers=True)
sns.lineplot(x='cat_smooth', y='mean f1 score', data=df, ax=axes[1], 
    markers=True)
fig.savefig('img/GS_GRDBST_catsmooth.png')
# best prarams {best prarams {'cat_smooth': 0}}
