#%%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import utils

#%%
# prepare data
x,y = utils.load_data_as_df('dataset/train.data')
smote = SMOTE(sampling_strategy=0.5, random_state=100)
x, y = smote.fit_sample(x, y)

#%%
# tuning parameters
# C是加在损失函数前面的，而不是加在正则项前面，因此C越大，表示对系数的惩罚力度越小，模型越容易过拟合
parameters = {'C':[0.1, 1, 5, 10, 13, 15, 20, 25, 30]}
# parameters = {'C':[0.1, 1, 5, 10]}
k = 5
clf = GridSearchCV(LogisticRegression(random_state=100, max_iter=500), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

# %%
# analysis result
results = search.cv_results_
print('best prarams', search.best_params_)
print('mean_fit_time', results['mean_fit_time'])
print('mean_test_score', results['mean_test_score'])
print('rank test score')
for i in results['rank_test_score']: # i is start with 1
    print('\tscore = {}\tparams = {}\tindex = {}'.format( results['mean_test_score'][i-1], results['params'][i-1], i))


df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'C': [ param_dic['C'] for param_dic in results['params'] ]
    }
    )

#%%
# fit time vs. mean_test_score
a4_dims = (20, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='mean fit time', y='mean f1 score', data=df, ax=ax)

for index, row in df.iterrows():
    ax.text(row['mean fit time'], row['mean f1 score'], row['C'], fontsize=16)

fig.savefig('img/GS_Linear_Regression_TIME_F1.png')

#%%
# c vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='C', y='mean f1 score',data=df, ax=ax)

for index, row in df.iterrows():
    ax.text(row['C'], row['mean f1 score'], row['C'], fontsize=16)

fig.savefig('img/GS_Linear_Regression_C_F1.png')

'''
output
=========

best prarams {'C': 10}
mean_fit_time [0.06736422 0.12396936 0.19409256 0.15260954]
mean_test_score [0.40485149 0.4117799  0.41289911 0.41427345]
rank test score
        score = 0.4142734537189764      params = {'C': 10}      index = 4
        score = 0.41289911294740056     params = {'C': 5}       index = 3
        score = 0.4117798961869415      params = {'C': 1}       index = 2
        score = 0.4048514921469984      params = {'C': 0.1}     index = 1
'''