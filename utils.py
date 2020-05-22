import pandas as pd
import seaborn  as sns; sns.set()
import matplotlib.pyplot as plt

def load_data(path:str):
    '''
        return numpy without feature 0 since it is high correlated to feature 1
    '''
    data_frame = pd.read_table(path, header=None, delim_whitespace=True)
    np_data = data_frame.loc[:, 1:].to_numpy()
    x = np_data[:, :-1]
    y = np_data[:,-1].reshape((-1,1))
    return (x,y)

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