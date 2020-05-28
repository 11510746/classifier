#%%
from sklearn.model_selection import GridSearchCV
from sklearn import svm
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
    'C':[0.01, 0.1, 1, 10, 50,100, 120,150,200],
    'gamma':['scale', 0.1, 0.01, 0.001]
    }
k = 5
clf = GridSearchCV(svm.SVC(kernel='rbf', random_state=10), parameters, scoring='f1', n_jobs=-1, cv=k)
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

# %%

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'C': [ param_dic['C'] for param_dic in results['params'] ],
        'gamma': [ param_dic['gamma'] for param_dic in results['params'] ],
    }
    )

#%%
# c vs. fit time
a4_dims = (20, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='C', y='mean fit time', hue='gamma', data=df, ax=ax, 
    markers=True, style='gamma')

fig.savefig('img/GS_SVM_C_TIME.png')

#%%
# c vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='C', y='mean f1 score', hue='gamma', data=df, ax=ax, 
    markers=True,  style='gamma')

fig.savefig('img/GS_SVM_C_F1.png')

'''
output
========================

best prarams {'C': 200, 'gamma': 'scale'}
mean_fit_time [ 5.54140134  6.33265333  4.87422605  4.10270514  4.3732913   4.91227283
  5.24745302  4.95296979  5.64329658  5.2007802   4.93963647  4.83928776
  7.48810706  6.28926277  5.2386744   4.79058857 10.74624634  6.15790677
  5.05383267  4.94783001 14.28070297  6.76687083  5.1728806   4.93161855
 15.63167691  6.61125355  5.01961589  5.21377096 22.81250305  8.87517667
  6.31052647  6.72396855 21.68611221  8.86173701  5.4753161   4.99492931]
mean_test_score [0.33920377 0.         0.         0.         0.39728154 0.09191091
 0.         0.         0.40968001 0.25751794 0.         0.
 0.41324107 0.26926458 0.17048531 0.         0.42200074 0.25382704
 0.23125179 0.17527852 0.42739406 0.26221449 0.25105953 0.22971159
 0.42826802 0.26657162 0.2534384  0.2320732  0.42923324 0.2714792
 0.25389839 0.23252218 0.43238521 0.28730562 0.25650673 0.22880432]
rank test score
        score = 0.3392037664689721      params = {'C': 0.01, 'gamma': 'scale'}  index = 0
        score = 0.0     params = {'C': 0.01, 'gamma': 0.1}      index = 1
        score = 0.0     params = {'C': 0.01, 'gamma': 0.01}     index = 2
        score = 0.0     params = {'C': 0.01, 'gamma': 0.001}    index = 3
        score = 0.39728153903687796     params = {'C': 0.1, 'gamma': 'scale'}   index = 4
        score = 0.0919109087435052      params = {'C': 0.1, 'gamma': 0.1}       index = 5
        score = 0.0     params = {'C': 0.1, 'gamma': 0.01}      index = 6
        score = 0.0     params = {'C': 0.1, 'gamma': 0.001}     index = 7
        score = 0.409680013115025       params = {'C': 1, 'gamma': 'scale'}     index = 8
        score = 0.257517939637785       params = {'C': 1, 'gamma': 0.1} index = 9
        score = 0.0     params = {'C': 1, 'gamma': 0.01}        index = 10
        score = 0.0     params = {'C': 1, 'gamma': 0.001}       index = 11
        score = 0.4132410698737992      params = {'C': 10, 'gamma': 'scale'}    index = 12
        score = 0.2692645799920615      params = {'C': 10, 'gamma': 0.1}        index = 13
        score = 0.1704853125100688      params = {'C': 10, 'gamma': 0.01}       index = 14
        score = 0.0     params = {'C': 10, 'gamma': 0.001}      index = 15
        score = 0.42200073971591456     params = {'C': 50, 'gamma': 'scale'}    index = 16
        score = 0.2538270394581768      params = {'C': 50, 'gamma': 0.1}        index = 17
        score = 0.23125178502688676     params = {'C': 50, 'gamma': 0.01}       index = 18
        score = 0.17527852392789028     params = {'C': 50, 'gamma': 0.001}      index = 19
        score = 0.42739406385964085     params = {'C': 100, 'gamma': 'scale'}   index = 20
        score = 0.2622144864603019      params = {'C': 100, 'gamma': 0.1}       index = 21
        score = 0.25105953337204306     params = {'C': 100, 'gamma': 0.01}      index = 22
        score = 0.22971158753958262     params = {'C': 100, 'gamma': 0.001}     index = 23
        score = 0.42826802383475704     params = {'C': 120, 'gamma': 'scale'}   index = 24
        score = 0.2665716184191343      params = {'C': 120, 'gamma': 0.1}       index = 25
        score = 0.25343840240764887     params = {'C': 120, 'gamma': 0.01}      index = 26
        score = 0.23207319806374443     params = {'C': 120, 'gamma': 0.001}     index = 27
        score = 0.42923324361188764     params = {'C': 150, 'gamma': 'scale'}   index = 28
        score = 0.2714792001349292      params = {'C': 150, 'gamma': 0.1}       index = 29
        score = 0.25389839288592725     params = {'C': 150, 'gamma': 0.01}      index = 30
        score = 0.23252218114939724     params = {'C': 150, 'gamma': 0.001}     index = 31
        score = 0.43238520512702133     params = {'C': 200, 'gamma': 'scale'}   index = 32
        score = 0.287305616203542       params = {'C': 200, 'gamma': 0.1}       index = 33
        score = 0.2565067326917599      params = {'C': 200, 'gamma': 0.01}      index = 34
        score = 0.22880431992446315     params = {'C': 200, 'gamma': 0.001}     index = 35
'''