#%%
import pandas as pd
import os
from loguru import logger
import seaborn  as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
#%%
def init_logger():
    os.mkdir('Log/', 775)
    logger.add('Log/{time}.log')

def load_data(path:str):
    '''
        return a DataFrame object, given the path
        return None if the path does not exist
    '''
    try:
        data_frame = pd.read_table(path, header=None, delim_whitespace=True)
    except pandas.errors.ParserError:
        logger.warning(path + ' DOES NOT EXIST!')
        return None

    return data_frame.loc[:, 1:] # remove feature 0 since it is high correlated to feature 1

#%%

def plot_training_cost(cost:list,save_path=''):
    plt.clf()
    df = pd.DataFrame( [(i, cost[i]) for i in range(len(cost))], columns=['iteration No.', 'cost'] )
    sns_plot = sns.lineplot(x='iteration No.', y='cost', data=df)
    if save_path:
        sns_plot.get_figure().savefig(save_path)

# def plot_fit(x, y, y_pred):
#     sns.scatterplot()
    # plt.scatter(x,y)
    # plt.plot(x, y_pred)
    # return 

#%%
if __name__ == '__main__':
    # init_logger()
    train_dframe = load_data('dataset/train.data')
    test_dframe = load_data('dataset/test.data')

    np_train, np_test = train_dframe.to_numpy(), test_dframe.to_numpy()
    x_train, y_train = np_train[:, :-1], np_train[:,-1].reshape((-1,1))
    x_test, y_test = np_test[:, :-1], np_test[:,-1].reshape((-1,1))

    # smote
    smote = SMOTE(sampling_strategy=0.5, random_state=100)
    x_train, y_train = smote.fit_sample(x_train, y_train)
    print(Counter(y_train))

    # SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))

# %%
print(load_data('dataset/test.data'))
