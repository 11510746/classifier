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

# prepare data
x,y = utils.load_data_as_df('dataset/train.data')
smote = SMOTE(sampling_strategy=0.5, random_state=100)
x, y = smote.fit_sample(x, y)


# tuning parameters
dt_list = []

parameters = {
    'n_estimators':[50, 100, 150, 200],  # number of trees, default = 100
    'learning_rate': [0.1, 1, 2, 3, 4, 5, 6, 7, 8,10]
    }

k = 5
clf = GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), random_state=10), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

# %%
# analysis result
results = search.cv_results_
print('best prarams', search.best_params_)
print('mean_fit_time', results['mean_fit_time'])
print('mean_test_score', results['mean_test_score'])
print('rank test score')
for i in range(len(results['rank_test_score'])): # i is start with 1
    print('\tscore = {}\tparams = {}\tindex = {}'.format( results['mean_test_score'][i], results['params'][i], i))

# %%

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'n_estimators': [ param_dic['n_estimators'] for param_dic in results['params'] ],
        'learning_rate': [ param_dic['learning_rate'] for param_dic in results['params'] ]
    }
    )

#%%
# variables vs. fit time
a4_dims = (20, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='learning_rate', y='mean fit time', hue='n_estimators', data=df, ax=ax, 
    markers=True, style='n_estimators')

fig.savefig('img/GS_AdaBoost_rate_time_d1.png')
#%%
# variables vs. mean_score
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(x='learning_rate', y='mean f1 score', hue='n_estimators', data=df, ax=ax, 
    markers=True, style='n_estimators')

fig.savefig('img/GS_AdaBoost_rate_F1_d1.png')


'''
output 
========================

best prarams {'learning_rate': 5, 'n_estimators': 50}
mean_fit_time [0.34571476 0.72437468 1.06253352 1.37477636 0.33224463 0.65541897
 0.97475824 1.39840326 0.35245051 0.73681216 1.20537744 1.70162373
 0.39392776 0.81935997 1.1277257  1.73115101 0.4077785  0.75182109
 1.39199476 1.60408144 0.3063478  0.46539083 0.79111314 1.12225499
 0.29725623 0.49889426 0.81379685 1.16186862 0.29067016 0.4803267
 0.78954425 0.94219542]
mean_test_score [0.45867692 0.46394956 0.46500792 0.46626409 0.46346685 0.46715218
 0.46839315 0.47006189 0.46407218 0.47319503 0.47264425 0.47390304
 0.45634912 0.45323401 0.46320085 0.46727797 0.4719266  0.47570455
 0.4796471  0.48419829 0.499968   0.499968   0.499968   0.499968
 0.25838623 0.25838623 0.25838623 0.25838623 0.25766738 0.25766738
 0.25766738 0.25766738]
rank test score
        score = 0.4586769249523022      params = {'learning_rate': 0.1, 'n_estimators': 50}     index = 0
        score = 0.4639495620428117      params = {'learning_rate': 0.1, 'n_estimators': 100}    index = 1
        score = 0.46500792198719826     params = {'learning_rate': 0.1, 'n_estimators': 150}    index = 2
        score = 0.46626409049635276     params = {'learning_rate': 0.1, 'n_estimators': 200}    index = 3
        score = 0.4634668456072264      params = {'learning_rate': 0.2, 'n_estimators': 50}     index = 4
        score = 0.46715218327161095     params = {'learning_rate': 0.2, 'n_estimators': 100}    index = 5
        score = 0.46839314659535003     params = {'learning_rate': 0.2, 'n_estimators': 150}    index = 6
        score = 0.4700618856712063      params = {'learning_rate': 0.2, 'n_estimators': 200}    index = 7
        score = 0.46407217622380303     params = {'learning_rate': 0.5, 'n_estimators': 50}     index = 8
        score = 0.4731950325182564      params = {'learning_rate': 0.5, 'n_estimators': 100}    index = 9
        score = 0.47264424987021697     params = {'learning_rate': 0.5, 'n_estimators': 150}    index = 10
        score = 0.4739030410479283      params = {'learning_rate': 0.5, 'n_estimators': 200}    index = 11
        score = 0.4563491229645993      params = {'learning_rate': 0.8, 'n_estimators': 50}     index = 12
        score = 0.45323401452988954     params = {'learning_rate': 0.8, 'n_estimators': 100}    index = 13
        score = 0.46320084600393996     params = {'learning_rate': 0.8, 'n_estimators': 150}    index = 14
        score = 0.46727797412626637     params = {'learning_rate': 0.8, 'n_estimators': 200}    index = 15
        score = 0.47192660447591284     params = {'learning_rate': 1, 'n_estimators': 50}       index = 16
        score = 0.4757045493339275      params = {'learning_rate': 1, 'n_estimators': 100}      index = 17
        score = 0.4796471012745753      params = {'learning_rate': 1, 'n_estimators': 150}      index = 18
        score = 0.48419828782137353     params = {'learning_rate': 1, 'n_estimators': 200}      index = 19
        score = 0.4999680000065473      params = {'learning_rate': 5, 'n_estimators': 50}       index = 20
        score = 0.4999680000065473      params = {'learning_rate': 5, 'n_estimators': 100}      index = 21
        score = 0.4999680000065473      params = {'learning_rate': 5, 'n_estimators': 150}      index = 22
        score = 0.4999680000065473      params = {'learning_rate': 5, 'n_estimators': 200}      index = 23
        score = 0.2583862272628512      params = {'learning_rate': 10, 'n_estimators': 50}      index = 24
        score = 0.2583862272628512      params = {'learning_rate': 10, 'n_estimators': 100}     index = 25
        score = 0.2583862272628512      params = {'learning_rate': 10, 'n_estimators': 150}     index = 26
        score = 0.2583862272628512      params = {'learning_rate': 10, 'n_estimators': 200}     index = 27
        score = 0.2576673771546715      params = {'learning_rate': 15, 'n_estimators': 50}      index = 28
        score = 0.2576673771546715      params = {'learning_rate': 15, 'n_estimators': 100}     index = 29
        score = 0.2576673771546715      params = {'learning_rate': 15, 'n_estimators': 150}     index = 30
        score = 0.2576673771546715      params = {'learning_rate': 15, 'n_estimators': 200}     index = 31

'''