#%%
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import os
from loguru import logger

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#%%
import utils

'''
Train classifiers with train.data (select parameter using cross validation in other files)
Test on test.data

evaluation: https://blog.csdn.net/sinat_26917383/article/details/75199996
            https://www.zhihu.com/question/30643044/answer/510317055
'''

logger.add('scores.log')

SMOT_RATIO = 0.5

x_train, y_train = utils.load_data('dataset/train.data')
x_test, y_test = utils.load_data('dataset/test.data')

# smote
smote = SMOTE(sampling_strategy=SMOT_RATIO, random_state=100)
x_train, y_train = smote.fit_sample(x_train, y_train)

# classifiers
classifiers = {
    'lg' : LogisticRegression(random_state=10, max_iter=1000, C = 15, solver = 'sag'),
    'lsvc' : LinearSVC(max_iter=100000,C=10),
    'svc' : SVC(kernel='rbf', random_state=10, C=500,gamma='scale'),
    'knn' : KNeighborsClassifier(n_neighbors = 5),
    'rf' : RandomForestClassifier(random_state=10, n_estimators=11, max_depth = 13, 
                min_samples_split = 30,min_samples_leaf = 20, max_features=0.8),
    'ada': AdaBoostClassifier(random_state=10, learning_rate = 0.001, n_estimators = 1),
    'gradboost' : GradientBoostingClassifier(learning_rate=1, n_estimators=80, max_depth=7,
                        min_samples_split=100),
    'xgb': XGBClassifier(max_depth=9, min_child_weight=1, learning_rate=0.1, 
                    n_estimators=220, colsample_bytree=1, subsample=0.6, reg_alpha=1, reg_lambda=1),
    'lgb': LGBMClassifier(max_depth=8, num_leaves=40, min_child_samples=18, min_child_weight=0.001,
                feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=3, cat_smooth=0)
} 


# %%
scores = {'accuracy':[], 'micro_precision':[], 'macro_precision':[], 'macro_recall':[],
            'micro_recall':[], 'f1':[]}
index = []

for name,clf in classifiers.items():
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred) # 每个类别求准确后，再求微平均
    scores['accuracy'].append(acc) 
    mac_pre = metrics.precision_score(y_test, y_pred, average='micro') # unweighted
    scores['micro_precision'].append(mac_pre) 
    mic_pre = metrics.precision_score(y_test, y_pred, average='macro')
    scores['macro_precision'].append(mic_pre)
    mac_recall = metrics.recall_score(y_test,y_pred, average='macro')
    scores['macro_recall'].append(mac_recall)
    mic_recall = metrics.recall_score(y_test,y_pred,average='micro')
    scores['micro_recall'].append(mic_recall)
    f1 = metrics.f1_score(y_test,y_pred,average='weighted')
    scores['f1'].append(f1)
    index.append(name)

    logger.warning(f'====================[{name}]====================')
    logger.info(f'\naccuracy={acc}\tweighted_f1={f1}\n'
                f'micro_precision={mic_pre}\tmac_precision={mac_pre}\n'
                f'mac_recall={mac_recall}\tmicro_recall={mic_recall}\n'
                )
    logger.info(f'classification report\n{metrics.classification_report(y_test, y_pred)}')
    logger.info(f'confusion matrix\n{metrics.confusion_matrix(y_test, y_pred, labels=[-1, 1])}')


scores_df = pd.DataFrame(scores)
scores_df.index = index
#%%
fig, axes = plt.subplots(3,2,figsize=(30,24))
sns.lineplot(y='accuracy', x=scores_df.index, data=scores_df, ax=axes[0][0])
sns.lineplot(y='f1', x=scores_df.index, data=scores_df, ax=axes[0][1])
sns.lineplot(y='micro_precision', x=scores_df.index, data=scores_df, ax=axes[1][0])
sns.lineplot(y='macro_precision', x=scores_df.index, data=scores_df, ax=axes[1][1])
sns.lineplot(y='macro_recall', x=scores_df.index, data=scores_df, ax=axes[2][0])
sns.lineplot(y='micro_recall', x=scores_df.index, data=scores_df, ax=axes[2][1])

fig.savefig('scores.png')

# %%
# confuse_dic = {
#     '-1':[1770,1787,1758,1616,1758,1387,1658,1713,1752],
#     '1':[44,42,46,24,39,75,26,27,37]
# }

# confuse_df = pd.DataFrame(confuse_dic)
# confuse_df.index = ['lg','lsvc','svc','knn','rf','ada','gradboost','xgb','lgb']
# fig, axes = plt.subplots(2,1,figsize=(30,16))
# sns.lineplot(x=confuse_df.index, y='-1',data=confuse_df,ax=axes[0],markers=True)
# sns.lineplot(x=confuse_df.index, y='1',data=confuse_df,ax=axes[1],markers=True)
# fig.savefig('confuse.png')


# %%
