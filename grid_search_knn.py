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

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'n_neighbors': [ param_dic['n_neighbors'] for param_dic in results['params'] ]
    }
    )

#%%
# k vs. mean_score
fig, ax = plt.subplots(figsize=(20,10))
sns.lineplot(x='n_neighbors', y='mean f1 score',data=df, ax=ax)

fig.savefig('img/GS_KNN_k_F1.png')

'''
output
=========

best prarams {'n_neighbors': 5}
'''