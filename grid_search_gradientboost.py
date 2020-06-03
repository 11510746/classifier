#%%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import utils

k=5
# prepare data
x,y = utils.load_data_as_df('dataset/train.data')
smote = SMOTE(sampling_strategy=0.5, random_state=100)
x, y = smote.fit_sample(x, y)

#%% n_estimators
parameters1 = {
    'n_estimators': range(20, 121, 20),
    'learning_rate': [0.01, 0.1, 0.5, 1]
    }
gsearch = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=parameters1, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)
#%%
df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'n_estimators': [ param_dic['n_estimators'] for param_dic in results['params'] ],
        'learning_rate': [ param_dic['learning_rate'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='n_estimators', y='mean fit time', hue='learning_rate', data=df, ax=axes[0], 
    markers=True, style='learning_rate')
sns.lineplot(x='n_estimators', y='mean f1 score', hue='learning_rate', data=df, ax=axes[1], 
    markers=True, style='learning_rate')
fig.savefig('img/GS_GRDBST_estimator_learnrate.png')


# best prarams {'learning_rate': 1, 'n_estimators': 80}

# %% max_depth
parameters2 = {
    'max_depth':range(3,14,2), 
    'min_samples_split':range(100,801,200)
}
gsearch = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=80,learning_rate=1), param_grid=parameters2, scoring='f1', n_jobs=-1, cv=k)
gsearch = gsearch.fit(x,y)
results = gsearch.cv_results_
print('best prarams', gsearch.best_params_)

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'max_depth': [ param_dic['max_depth'] for param_dic in results['params'] ],
        'min_samples_split': [ param_dic['min_samples_split'] for param_dic in results['params'] ]
    }
    )

fig,axes = plt.subplots(2,1,figsize=(30,16))
sns.lineplot(x='max_depth', y='mean fit time', hue='min_samples_split', data=df, ax=axes[0], 
    markers=True, style='min_samples_split')
sns.lineplot(x='max_depth', y='mean f1 score', hue='min_samples_split', data=df, ax=axes[1], 
    markers=True, style='min_samples_split')
fig.savefig('img/GS_GRDBST_maxdepth_minsample.png')
# best prarams {'max_depth': 7, 'min_samples_split': 100}

