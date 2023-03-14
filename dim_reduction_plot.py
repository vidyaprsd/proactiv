import os.path
import nptsne
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
from params import get_params

def compute_tsne_embedding(df, start_d_index, end_d_index):
    '''
    compute the tsne embedding based on difference values
    '''
    for i in range(start_d_index, end_d_index):
        if i == start_d_index:
            df['combined']          = df[df.columns[i]]
        else:
            df['combined']         += df[df.columns[i]]

    values                          = df['combined'].values.tolist()
    scaler                          = MinMaxScaler(copy=False)
    scaler.fit_transform(values)

    tsneobj                         = nptsne.TextureTsne(False, 1000, 2, 20, 800, nptsne.KnnAlgorithm.Flann)
    embedding                       = tsneobj.fit_transform(values)
    embedding                       = embedding.reshape(int(embedding.shape[0]/2), 2)
    return embedding

def plot_transform_embedding(embedding, df, colorby_funcs, name, cmap='viridis'):
    ''''
    Plot the (optimization) transform summary
    '''
    fig, ax                         = plt.subplots(1, len(colorby_funcs), figsize=(15, 5))
    plots                           = []

    for i in range(len(colorby_funcs)):
        colorvalues                 = []
        for key in df.groupby(colorby_funcs).groups.keys():
            colorvalues.append(list(key)[i])
        plots.append(ax[i].scatter(embedding[:, 0], embedding[:, 1], marker='o', cmap=cmap, c=colorvalues, alpha=0.7))
        ax[i].set_title("colored by: " + colorby_funcs[i])
        ax[i].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        fig.colorbar(plots[i], fraction=0.046, pad=0.04, ax=ax[i])

    fig.suptitle(name)
    plt.show()

def plot_instance_embedding(embedding, name):
    '''
    Plot the optimization instance summary
    '''
    plt.scatter(embedding[:, 0], embedding[:, 1], marker='o', alpha=0.7, c='grey')
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.title(name)
    plt.show()

if __name__ == '__main__':
    params                          = get_params()
    differences                     = np.array(pickle.load(open(os.path.join(params['logdir'], params['differences_fn']), 'rb')))
    differences_df                  = pd.DataFrame(differences)
    colnames                        = ['path']

    colnames.extend(params["transforms"]["types"])
    colnames.extend(['subid'])
    colnames.extend(range(differences.shape[1]-5))
    differences_df.iloc[:,1:]       = differences_df.iloc[:,1:].astype(float)
    differences_df.columns          = colnames

    # concatenate difference vector d per transform T across all instances x_i
    differences_per_transform       = differences_df.groupby(params["transforms"]["types"]).agg(lambda x: x.tolist())
    # concatenate difference vector d per transform T across all instances x_i
    differences_per_instance        = differences_df.groupby(['subid']).agg(lambda x: x.tolist())

    #reduce dimensionality with t-sne
    transform_tsne_embedding        = compute_tsne_embedding(differences_per_transform, 3, len(differences_per_transform.columns) - 4)
    instance_tsne_embedding         = compute_tsne_embedding(differences_per_instance, 4, len(differences_per_instance.columns) - 4)

    plot_transform_embedding(transform_tsne_embedding, differences_per_transform,
                             params["transforms"]["types"], "Optimization transform summary")

    plot_instance_embedding(instance_tsne_embedding, "Instance transform summary")






