import pandas as pd
import seaborn  as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np

def load_data(path:str):
    '''
        return numpy without feature 0 since it is high correlated to feature 1
    '''
    data_frame = pd.read_table(path, header=None, delim_whitespace=True)
    np_data = data_frame.loc[:, 1:].to_numpy()
    x = np_data[:, :-1]
    y = np_data[:,-1].reshape((-1,1))
    return (x,y)

def load_data_as_df(path:str):
    data_frame = pd.read_table(path, header=None, delim_whitespace=True)
    return (data_frame.loc[:, 1:len(data_frame.columns)-2], data_frame.loc[:, len(data_frame.columns)-1])

def plot_training_cost(cost:list,save_path=''):
    plt.clf()
    df = pd.DataFrame( [(i, cost[i]) for i in range(len(cost))], columns=['iteration No.', 'cost'] )
    sns_plot = sns.lineplot(x='iteration No.', y='cost', data=df)
    if save_path:
        sns_plot.get_figure().savefig(save_path)

def plot_line(x,y,save_name):
    plt.clf()
    plt.plot(x, y)

    # show annotation for the max point
    max_index = np.argmax(y)
    text = '[{}, {}]'.format(x[max_index], round(y[max_index],2))
    plt.annotate(text, xytext=(x[max_index], y[max_index]),
        xy=(x[max_index], y[max_index]))

    plt.savefig(save_name)

# %%
