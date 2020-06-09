#%%
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np

import seaborn  as sns; sns.set()
import matplotlib.pyplot as plt

import utils

'''
explore the impact of different smote ratio on the precision, recall for each class, and the accuracy
'''

#%%
if __name__ == "__main__":
    x,y = utils.load_data('dataset/train.data')
    k = 5

    param_ratio = [0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1]

    precision_1 = [0 for i in range(len(param_ratio))]
    precision_2 = [0 for i in range(len(param_ratio))]
    recall_1 = [0 for i in range(len(param_ratio))]
    recall_2 = [0 for i in range(len(param_ratio))]
    accuracy = [0 for i in range(len(param_ratio))]

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)

    for train_index, test_index in kf.split(x,y):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        for i in range(len(param_ratio)):
            r = param_ratio[i]
            smote = None
            sm_x_train = x_train
            sm_y_train = y_train

            print('ratio =', r)

            if r > 0:
                smote = SMOTE(sampling_strategy=r, random_state=100)
                sm_x_train, sm_y_train = smote.fit_sample(x_train, y_train)
                print(Counter(sm_y_train))

            clf = svm.SVC(kernel='linear')
            clf.fit(sm_x_train, sm_y_train)
            y_pred = clf.predict(x_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            precision_1[i] += report['1.0']['precision']
            precision_2[i] += report['-1.0']['precision']
            recall_1[i] += report['1.0']['recall']
            recall_2[i] += report['-1.0']['recall']
            accuracy[i] += report['accuracy']

    precision_1 = np.array(precision_1) * 100 / k
    precision_2 = np.array(precision_2) * 100 / k
    recall_1 = np.array(recall_1) * 100 / k
    recall_2 = np.array(recall_2) * 100 / k
    accuracy = np.array(accuracy) * 100 / k

    fig,axes = plt.subplots(2,2,figsize=(30,16))
    axes[0][0].set_xlabel('sampling_strategy')
    axes[0][0].set_ylabel('precision for class 1')
    axes[0][1].set_xlabel('sampling_strategy')
    axes[0][1].set_ylabel('precision for class -1')
    axes[1][0].set_xlabel('sampling_strategy')
    axes[1][0].set_ylabel('recall for class 1')
    axes[1][1].set_xlabel('sampling_strategy')
    axes[1][1].set_ylabel('recall for class -1')
    sns.lineplot(x=param_ratio, y=precision_1,markers=True,ax=axes[0][0])
    sns.lineplot(x=param_ratio, y=precision_2,markers=True,ax=axes[0][1])
    sns.lineplot(x=param_ratio, y=recall_1,markers=True,ax=axes[1][0])
    sns.lineplot(x=param_ratio, y=recall_2,markers=True,ax=axes[1][1])
    fig.savefig('img/SM_PRE_RECALL.png')

#%%
    fig,ax = plt.subplots()
    ax.set_xlabel('sampling_strategy')
    ax.set_ylabel('accuracy')
    sns.lineplot(x=param_ratio, y=accuracy,markers=True,ax=ax)
    fig.savefig('img/SM_AC.png')




# %%
