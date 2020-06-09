#%%
import seaborn  as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

'''
1. Print min and max for each feature
2. plot the correlation of features
3. count num of samples for each class
'''
#%%
dframe = pd.read_table('dataset/train.data', header=None, delim_whitespace=True)

#%%
# min and max
# print('Min val for each feature')
# print(dframe.min())
# print('Max val for each feature')
# print(dframe.max())


#%%
# correlation of feature
# plt.clf()
# corr = dframe.corr().round(2)
# mask = np.tril(np.ones(corr.shape)).astype(np.bool)
# corr_lt = corr.where(mask) # only keep lower triangle
# sns_plot = sns.heatmap(corr_lt, vmin=0, vmax=1, annot=True, cmap="Blues")
# sns_plot.get_figure().savefig('img/feature_corr.png')

# %%
# num of samples for each class
print('No. of samples in each class')
print(Counter(dframe[len(dframe.columns)-1]))



# %%
