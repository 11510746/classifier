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

# tuning parameters
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10, 20, 30]}
k = 5
clf = GridSearchCV(svm.SVC(gamma='scale'), parameters, scoring='f1', n_jobs=-1, cv=k)
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

# classify params by kernel
label_list = []
for i in range( len(results['params']) ):
    if results['params'][i]['kernel'] == 'linear':
        label_list.append('kernel = linear')
    else:
        label_list.append('kernel = rbf')

# %%

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'C': [ param_dic['C'] for param_dic in results['params'] ],
        'label': label_list
    }
    )

#%%
# fit time vs. mean_test_score
a4_dims = (20, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='mean fit time', y='mean f1 score', hue='label', data=df, ax=ax, 
    markers=True, style='label')

for index, row in df.iterrows():
    ax.text(row['mean fit time'], row['mean f1 score'] + 0.003, index, fontsize=16)

fig.savefig('img/GS_SVM_TIME_F1.png')

#%%
# c vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='C', y='mean f1 score', hue='label', data=df, ax=ax, 
    markers=True, style='label')

for index, row in df.iterrows():
    ax.text(row['C'], row['mean f1 score'], index, fontsize=16)

fig.savefig('img/GS_SVM_C_F1.png')

'''
output - gamma = 'scale'
========================

best prarams {'C': 30, 'kernel': 'rbf'}
mean_fit_time [3.4062192  4.83093104 5.69669056 5.22683425 3.72070274 5.62876949
 6.04761415 6.69517283 7.91610894 7.47114401]
mean_test_score [0.25206972 0.40968001 0.28603026 0.4104132  0.28571048 0.41324107
 0.28588488 0.4182845  0.28874178 0.41948104]
rank test score
        score = 0.4194810372473414      params = {'C': 30, 'kernel': 'rbf'}     index = 10
        score = 0.28571048453386855     params = {'C': 10, 'kernel': 'linear'}  index = 5
        score = 0.28588487754170255     params = {'C': 20, 'kernel': 'linear'}  index = 7
        score = 0.4104132041767968      params = {'C': 5, 'kernel': 'rbf'}      index = 4
        score = 0.28874177769264137     params = {'C': 30, 'kernel': 'linear'}  index = 9
        score = 0.28603026234562        params = {'C': 5, 'kernel': 'linear'}   index = 3
        score = 0.41828450168953946     params = {'C': 20, 'kernel': 'rbf'}     index = 8
        score = 0.409680013115025       params = {'C': 1, 'kernel': 'rbf'}      index = 2
        score = 0.4132410698737992      params = {'C': 10, 'kernel': 'rbf'}     index = 6
        score = 0.25206972123106863     params = {'C': 1, 'kernel': 'linear'}   index = 1

output - gamma = 'auto'
========================

best prarams {'C': 30, 'kernel': 'linear'}
mean_fit_time [ 4.80212283  5.53120656  4.9474226   6.27612238  5.85611949  6.007514
  9.21495681  7.27211714 12.27039342  6.5236536 ]
mean_test_score [0.25206972 0.25753324 0.28603026 0.27010394 0.28571048 0.27036987
 0.28588488 0.26302352 0.28874178 0.25571761]
rank test score
        score = 0.2557176149069844      params = {'C': 30, 'kernel': 'rbf'}     index = 10
        score = 0.2630235249993632      params = {'C': 20, 'kernel': 'rbf'}     index = 8
        score = 0.2575332421953931      params = {'C': 1, 'kernel': 'rbf'}      index = 2
        score = 0.27036986549210373     params = {'C': 10, 'kernel': 'rbf'}     index = 6
        score = 0.2701039407190138      params = {'C': 5, 'kernel': 'rbf'}      index = 4
        score = 0.28571048453386855     params = {'C': 10, 'kernel': 'linear'}  index = 5
        score = 0.28603026234562        params = {'C': 5, 'kernel': 'linear'}   index = 3
        score = 0.28588487754170255     params = {'C': 20, 'kernel': 'linear'}  index = 7
        score = 0.25206972123106863     params = {'C': 1, 'kernel': 'linear'}   index = 1
        score = 0.28874177769264137     params = {'C': 30, 'kernel': 'linear'}  index = 9
'''