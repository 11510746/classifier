#%%
from sklearn.model_selection import KFold
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

def plot_line(x,y,save_name):
    plt.clf()
    plt.plot(x, y)

    # show annotation for the max point
    max_index = np.argmax(y)
    text = '[{}, {}]'.format(x[max_index], round(y[max_index],2))
    plt.annotate(text, xytext=(x[max_index], y[max_index]),
        xy=(x[max_index], y[max_index]))

    plt.savefig(save_name)

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

    for i in range(len(param_ratio)):
        r = param_ratio[i]

        smote = None
        sm_x_train = x
        sm_y_train = y

        print('ratio =', r)
        if r > 0:
            smote = SMOTE(sampling_strategy=r, random_state=100)
            sm_x_train, sm_y_train = smote.fit_sample(x_train, y_train)
            print(Counter(sm_y_train)) 

        kf = KFold(n_splits=k, shuffle=True, random_state=10)

        for train_index, test_index in kf.split(sm_x_train):
            x_train, y_train = sm_x_train[train_index], sm_y_train[train_index]
            x_test, y_test = sm_x_train[test_index], sm_y_train[test_index]

            clf = svm.SVC(kernel='linear', random_state=10)
            clf.fit(x_train, y_train)
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

    plot_line(param_ratio, precision_1, 'img/_SM_PRE_1.png')
    plot_line(param_ratio, precision_2, 'img/_SM_PRE_2.png')
    plot_line(param_ratio, recall_1, 'img/_SM_RC_1.png')
    plot_line(param_ratio, recall_2, 'img/_SM_RC_2.png')
    plot_line(param_ratio, accuracy, 'img/_SM_AC.png')


