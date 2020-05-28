#%%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import utils

# prepare data
x,y = utils.load_data_as_df('dataset/train.data')
smote = SMOTE(sampling_strategy=0.5, random_state=100)
x, y = smote.fit_sample(x, y)

# tuning parameters
parameters = {
    'n_estimators':[50, 100, 150, 200],  # number of trees, default = 100
    'max_depth':[10, 20, 30, None], # default = None, util all leaves are pure or until all leaves contain less than min_samples_split samples. 数据少或者特征少的时候可以不管这个值
    }
k = 5
clf = GridSearchCV(RandomForestClassifier(random_state=10, n_jobs=-1), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

# %%
# analysis result
results = search.cv_results_
print('best prarams', search.best_params_)
print('mean_fit_time', results['mean_fit_time'])
print('mean_test_score', results['mean_test_score'])
for i in range(len(results['params'])):
    print('\tscore = {}\tparams = {}\tindex = {}'.format( results['mean_test_score'][i], results['params'][i], i))

# %%

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'n_estimators': [ param_dic['n_estimators'] for param_dic in results['params'] ],
        'max_depth': [ param_dic['max_depth'] if param_dic['max_depth'] else -1 for param_dic in results['params'] ],
    }
    )

print(df)

#%%
# variables vs. fit time
a4_dims = (20, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='n_estimators', y='mean fit time', hue='max_depth', data=df, ax=ax, 
    markers=True, style='max_depth')

fig.savefig('img/GS_RandomForest_estimators_time.png')
#%%
# variables vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='n_estimators', y='mean f1 score', hue='max_depth', data=df, ax=ax, 
    markers=True, style='max_depth')

fig.savefig('img/GS_RandomForest_estimators_F1.png')


'''
output 
========================

best prarams {'max_depth': 20, 'n_estimators': 200}
mean_fit_time [0.45845709 0.86850724 1.32159166 1.89447813 0.66899581 1.15405645
 1.83812723 2.43881793 0.78430185 1.41316381 2.04655852 2.73803267
 0.78861003 1.45904651 2.18311934 2.42239861]
mean_test_score [0.5349632  0.53785662 0.53409951 0.53192795 0.61721094 0.61801114
 0.61828277 0.61899603 0.60148339 0.59970006 0.5996615  0.60142896
 0.58375856 0.58600877 0.58593813 0.58565044]
        score = 0.5349631981283224      params = {'max_depth': 10, 'n_estimators': 50}  index = 0
        score = 0.5378566222475166      params = {'max_depth': 10, 'n_estimators': 100} index = 1
        score = 0.5340995056304799      params = {'max_depth': 10, 'n_estimators': 150} index = 2
        score = 0.5319279486727455      params = {'max_depth': 10, 'n_estimators': 200} index = 3
        score = 0.6172109368019576      params = {'max_depth': 20, 'n_estimators': 50}  index = 4
        score = 0.6180111403384541      params = {'max_depth': 20, 'n_estimators': 100} index = 5
        score = 0.6182827665679567      params = {'max_depth': 20, 'n_estimators': 150} index = 6
        score = 0.6189960255700445      params = {'max_depth': 20, 'n_estimators': 200} index = 7
        score = 0.6014833869686258      params = {'max_depth': 30, 'n_estimators': 50}  index = 8
        score = 0.599700055093438       params = {'max_depth': 30, 'n_estimators': 100} index = 9
        score = 0.5996614986627475      params = {'max_depth': 30, 'n_estimators': 150} index = 10
        score = 0.6014289580419663      params = {'max_depth': 30, 'n_estimators': 200} index = 11
        score = 0.5837585617399995      params = {'max_depth': None, 'n_estimators': 50}        index = 12
        score = 0.5860087715687587      params = {'max_depth': None, 'n_estimators': 100}       index = 13
        score = 0.5859381299523098      params = {'max_depth': None, 'n_estimators': 150}       index = 14
        score = 0.5856504392106036      params = {'max_depth': None, 'n_estimators': 200}       index = 15
    mean fit time  mean f1 score  n_estimators  max_depth
0        0.458457       0.534963            50         10
1        0.868507       0.537857           100         10
2        1.321592       0.534100           150         10
3        1.894478       0.531928           200         10
4        0.668996       0.617211            50         20
5        1.154056       0.618011           100         20
6        1.838127       0.618283           150         20
7        2.438818       0.618996           200         20
8        0.784302       0.601483            50         30
9        1.413164       0.599700           100         30
10       2.046559       0.599661           150         30
11       2.738033       0.601429           200         30
12       0.788610       0.583759            50         -1
13       1.459047       0.586009           100         -1
14       2.183119       0.585938           150         -1
15       2.422399       0.585650           200         -1
'''