#%%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
parameters = {'n_neighbors':[i for i in range(2,13)]}
k = 5
clf = GridSearchCV( KNeighborsClassifier(), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

# %%
# analysis result
results = search.cv_results_
print('best prarams', search.best_params_)
print('mean_fit_time', results['mean_fit_time'])
print('mean_test_score', results['mean_test_score'])
for i in range( len(results['params']) ):
    print('\tscore = {}\tparams = {}\tindex = {}'.format( results['mean_test_score'][i], results['params'][i], i))


df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'n_neighbors': [ param_dic['n_neighbors'] for param_dic in results['params'] ]
    }
    )

#%%
# k vs. fit time 
a4_dims = (20, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='n_neighbors', y='mean fit time',data=df, ax=ax)

fig.savefig('img/GS_KNN_k_TIME.png')

#%%
# k vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='n_neighbors', y='mean f1 score',data=df, ax=ax)

fig.savefig('img/GS_KNN_k_F1.png')

'''
output
=========

best prarams {'n_neighbors': 5}
mean_fit_time [0.14194026 0.14180527 0.14497991 0.1648592  0.15064807 0.15985832
 0.16027298 0.14642062 0.1714963  0.14605713 0.13597393]
mean_test_score [0.52254948 0.61589211 0.57352752 0.62176822 0.58611712 0.61692376
 0.59028798 0.61326882 0.58857557 0.61408995 0.59276645]
        score = 0.522549479389156       params = {'n_neighbors': 2}     index = 0
        score = 0.6158921069789611      params = {'n_neighbors': 3}     index = 1
        score = 0.5735275209738143      params = {'n_neighbors': 4}     index = 2
        score = 0.6217682213183733      params = {'n_neighbors': 5}     index = 3
        score = 0.5861171205061528      params = {'n_neighbors': 6}     index = 4
        score = 0.6169237625365755      params = {'n_neighbors': 7}     index = 5
        score = 0.5902879822861972      params = {'n_neighbors': 8}     index = 6
        score = 0.6132688167523048      params = {'n_neighbors': 9}     index = 7
        score = 0.5885755673942834      params = {'n_neighbors': 10}    index = 8
        score = 0.6140899494481326      params = {'n_neighbors': 11}    index = 9
        score = 0.5927664543781668      params = {'n_neighbors': 12}    index = 10
'''