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
k = 5
clf = GridSearchCV(LogisticRegression(random_state=10, max_iter=500), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

# %%
# analysis result
results = search.cv_results_
print('best prarams', search.best_params_)
print('mean_fit_time', results['mean_fit_time'])
print('mean_test_score', results['mean_test_score'])
for i in range(len(results['params'])):
    print('\tscore = {}\tparams = {}\tindex = {}'.format( results['mean_test_score'][i], results['params'][i], i))


df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'C': [ param_dic['C'] for param_dic in results['params'] ]
    }
    )

#%%
# c vs. fit time
a4_dims = (20, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot( x='C', y='mean fit time',data=df, ax=ax, markers=True)

fig.savefig('img/GS_Linear_Regression_TIME_F1.png')

#%%
# c vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='C', y='mean f1 score',data=df, ax=ax, markers=True)

fig.savefig('img/GS_Linear_Regression_C_F1.png')

'''
output
=========

best prarams {'C': 13}
mean_fit_time [0.06250534 0.15488591 0.20997124 0.25304446 0.34482269 0.24843531
 0.24614549 0.19422388 0.1872921 ]
mean_test_score [0.40485149 0.4117799  0.41291126 0.41434096 0.41468284 0.41440244
 0.4146033  0.41400132 0.41332463]
        score = 0.4048514921469984      params = {'C': 0.1}     index = 0
        score = 0.4117798961869415      params = {'C': 1}       index = 1
        score = 0.41291126474716133     params = {'C': 5}       index = 2
        score = 0.41434095692328654     params = {'C': 10}      index = 3
        score = 0.41468283726516686     params = {'C': 13}      index = 4
        score = 0.41440244324261577     params = {'C': 15}      index = 5
        score = 0.41460330077189556     params = {'C': 20}      index = 6
        score = 0.4140013207791521      params = {'C': 25}      index = 7
        score = 0.41332463301356154     params = {'C': 30}      index = 8
'''