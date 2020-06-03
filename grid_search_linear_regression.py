#%%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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
# C是加在损失函数前面的，而不是加在正则项前面，因此C越大，表示对系数的惩罚力度越小，模型越容易过拟合

params = {'C':[0.1, 1, 5, 10, 13, 15, 20, 25, 30],
          'solver':['liblinear','sag','lbfgs','newton-cg']
         }

k = 5
clf = GridSearchCV(LogisticRegression(random_state=10, max_iter=1000), params, scoring='f1', n_jobs=-1, cv=k)
search = clf.fit(x, y)

results = search.cv_results_
print('best prarams', search.best_params_)

#%%
df = pd.DataFrame(
    {
        'mean f1 score': results['mean_test_score'],
        'C': [ param_dic['C'] for param_dic in results['params'] ],
        'solver': [ param_dic['solver'] for param_dic in results['params'] ]
    }
    )

#%%
# c vs. solver
fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot( x='C', y='mean f1 score', style = 'solver', data=df, ax=ax, markers=True)

fig.savefig('img/GS_Linear_c_solver.png')

# best prarams {'C': 15, 'solver': 'sag'}