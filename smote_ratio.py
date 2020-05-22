from sklearn.model_selection import KFold, cross_val_score
import utils

if __name__ == "__main__":
    x,y = utils.load_data('dataset/train.data')
    k = 5

    param_ratio = [r for r in range(0.45, 0.6)]

    kf = KFold(n_splits=k, shuffle=True, random_state=10)
    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]


#%%
# import numpy as np

# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

