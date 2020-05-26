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
clf = GridSearchCV(svm.LinearSVC(), parameters, scoring='f1', n_jobs=-1, cv=k)
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

fig.savefig('img/GS_Linear_SVM_TIME_F1.png')

#%%
# c vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='C', y='mean f1 score',data=df, ax=ax)

for index, row in df.iterrows():
    ax.text(row['C'], row['mean f1 score'], row['C'], fontsize=16)

fig.savefig('img/GS_Linear_SVM_C_F1.png')

'''
output
=========

best prarams {'C': 13}
mean_fit_time [0.06799264 0.61439934 1.96887369 2.08146834 2.07369819 2.40032601
 3.11767888 2.54858017 2.73898482]
mean_test_score [0.40211462 0.40444709 0.40515327 0.40200299 0.40638823 0.39814775
 0.38550589 0.37019476 0.35520306]
rank test score
	score = 0.40200299229520864	params = {'C': 10}	index = 4
	score = 0.4051532675871899	params = {'C': 5}	index = 3
	score = 0.4044470860990284	params = {'C': 1}	index = 2
	score = 0.4063882309057127	params = {'C': 13}	index = 5
	score = 0.40211462244988916	params = {'C': 0.1}	index = 1
	score = 0.39814775109213224	params = {'C': 15}	index = 6
	score = 0.3855058901192209	params = {'C': 20}	index = 7
	score = 0.3701947552511363	params = {'C': 25}	index = 8
	score = 0.35520305854035	params = {'C': 30}	index = 9
'''