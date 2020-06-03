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

df = pd.DataFrame(
    {
        'mean fit time': results['mean_fit_time'],
        'mean f1 score': results['mean_test_score'],
        'C': [ param_dic['C'] for param_dic in results['params'] ]
    }
    )

#%%
fig, axes = plt.subplots(1,2,figsize=(20, 10))
sns.lineplot(x='C', y='mean fit time', data=df, ax=axes[0], markers=True)
sns.lineplot(x='C', y='mean fit time', data=df, ax=axes[1], markers=True)

fig.savefig('img/GS_Linear_SVM_c.png')

# best prarams {'C': 10}