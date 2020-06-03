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
    'C':[300, 400, 500, 600,700],
    'gamma':['scale', 0.1, 1]
    }
k = 5
clf = GridSearchCV(svm.SVC(kernel='rbf', random_state=10), parameters, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

# %%
# analysis result
results = search.cv_results_
print('best prarams', search.best_params_)

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
sns.lineplot(x='C', y='mean f1 score', data=df, ax=ax, 
    markers=True, style='gamma')
# fig.savefig('img/GS_SVM_rbf_gamma_scale.png')
fig.savefig('img/GS_SVM_rbf_c_gamma.png')

# best prarams {'C': 500, 'gamma': 'scale'}