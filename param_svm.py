'''
reference: https://xijunlee.github.io/2017/03/29/sklearn中SVM调参说明及经验总结

我们使用linear kernel, 只要讨论惩罚系数 C，即对误差的宽容度。
c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。
C过大或过小，泛化能力变差
'''

#%%
from sklearn.model_selection import cross_validate
from sklearn import svm

import utils


#%%
if __name__ == "__main__":
    x,y = utils.load_data('dataset/train.data')

    param_c = [0.1, 0.5, 1, 2, 5, 10, 20, 30]
    
    score_list = []
    scoring = ['accuracy', 'f1_weighted']

    for c in param_c:
        clf = svm.SVC(kernel='linear', C=c, random_state=0)
        score_list.append(cross_validate(clf, x, y, cv=5, scoring=scoring))

    accuracy_list = []
    f1_weighted_list = []

    for score in score_list:
        accuracy_list.append(score['test_accuracy'].mean())
        f1_weighted_list.append(score['test_f1_weighted'].mean())

    utils.plot_line(param_c, accuracy_list, 'img/PARAM_SVM_ACC.png')
    utils.plot_line(param_c, f1_weighted_list, 'img/PARAM_SVM_F1W.png')



# %%
    for i in range(len(param_c)):
        print('c={}, accuracy={}, f1_weighted={}'.format(param_c[i], 
            accuracy_list[i], f1_weighted_list[i]))

    # Output
    # c=0.1, accuracy=0.9437537718768858, f1_weighted=0.9164444697113368
    # c=0.5, accuracy=0.9437537718768858, f1_weighted=0.9164444697113368
    # c=1, accuracy=0.9437537718768858, f1_weighted=0.9164444697113368
    # c=2, accuracy=0.9437537718768858, f1_weighted=0.9164444697113368
    # c=5, accuracy=0.9437537718768858, f1_weighted=0.9164444697113368
    # c=10, accuracy=0.9437537718768858, f1_weighted=0.9164444697113368
    # c=20, accuracy=0.9437537718768858, f1_weighted=0.9164444697113368
    # c=30, accuracy=0.9437537718768858, f1_weighted=0.9164444697113368

# %%
