#%%
import os
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

import utils


#%%
def init_logger():
    os.mkdir('Log/', 775)
    logger.add('Log/{time}.log')


#%%
if __name__ == '__main__':
    # init_logger()

    x_train, y_train = utils.load_data('dataset/train.data')
    x_test, y_test = utils.load_data('dataset/test.data')

    # smote
    smote = SMOTE(sampling_strategy=0.5, random_state=100)
    x_train, y_train = smote.fit_sample(x_train, y_train)
    print(Counter(y_train))

    # SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
