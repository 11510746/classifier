#%%
from sklearn.model_selection import GridSearchCV
from sklearn import svm
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
parameters = {'C':[0.1, 1, 5, 10, 13, 15, 20, 25, 30]}
k = 5
clf = GridSearchCV(svm.LinearSVC(max_iter=100000), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

# %%
# analysis result
results = search.cv_results_
print('best prarams', search.best_params_)
print('mean_fit_time', results['mean_fit_time'])
print('mean_test_score', results['mean_test_score'])
print('rank test score')
for i in range( len(results['params']) ):
    print('\tscore = {}\tparams = {}\tindex = {}'.format( results['mean_test_score'][i], results['params'][i], i))


df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'C': [ param_dic['C'] for param_dic in results['params'] ]
    }
    )

#%%
# c vs.fit time
a4_dims = (20, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='C', y='mean fit time', data=df, ax=ax, markers=True)


fig.savefig('img/GS_Linear_SVM_TIME_F1.png')

#%%
# c vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='C', y='mean f1 score',data=df, ax=ax, markers=True)

fig.savefig('img/GS_Linear_SVM_C_F1.png')

'''
output
=========

best prarams {'C': 10}
mean_fit_time [ 0.04410419  0.53234153  2.87609258  6.17603197  8.16919713  9.03650932
 12.63729715 14.57709332 10.73044958]
mean_test_score [0.40211462 0.40444709 0.40515327 0.40563171 0.40562786 0.40562786
 0.40562786 0.40562786 0.40562786]
rank test score
        score = 0.40211462244988916     params = {'C': 0.1}     index = 0
        score = 0.4044470860990284      params = {'C': 1}       index = 1
        score = 0.4051532675871899      params = {'C': 5}       index = 2
        score = 0.4056317120791748      params = {'C': 10}      index = 3
        score = 0.4056278598952995      params = {'C': 13}      index = 4
        score = 0.4056278598952995      params = {'C': 15}      index = 5
        score = 0.4056278598952995      params = {'C': 20}      index = 6
        score = 0.4056278598952995      params = {'C': 25}      index = 7
        score = 0.4056278598952995      params = {'C': 30}      index = 8
'''