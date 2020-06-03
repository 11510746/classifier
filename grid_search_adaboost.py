#%%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import utils

# https://ask.hellobi.com/blog/zhangjunhong0428/12405

# prepare data
x,y = utils.load_data_as_df('dataset/train.data')
smote = SMOTE(sampling_strategy=0.5, random_state=100)
x, y = smote.fit_sample(x, y)
k = 5

best_params = {}
#%% estimators and learning rate

parameters = {
    'n_estimators':range(1,51,10),  # number of trees, default = 100
    'learning_rate': [0.001, 0.01, 0.1, 1]
    }


clf = GridSearchCV(AdaBoostClassifier(random_state=10), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)
results = search.cv_results_

print(search.best_params_)

best_params['n_estimators'] = search.best_params_['n_estimators']
best_params['learning_rate'] = search.best_params_['learning_rate']

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'n_estimators': [ param_dic['n_estimators'] for param_dic in results['params'] ],
        'learning_rate': [ param_dic['learning_rate'] for param_dic in results['params'] ]
    }
    )

fig, ax = plt.subplots(figsize=(20,10))
sns.lineplot(x='learning_rate', y='mean f1 score', hue='n_estimators',style='n_estimators', data=df, ax=ax, 
    markers=True)

fig.savefig('img/GS_AdaBoost_estimator_learnrate.png')

# best parameters {'learning_rate': 0.001, 'n_estimators': 1}


