from glob import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


instancesPath = "./instances/"
solutionsPath = "./lab2/solutions/"
outputPath = "./lab2/plots/"

colors = ['red', 'green', 'blue']


for instancePath in glob(os.path.join(instancesPath, "*.csv")):
    # Read raw Instance
    instance = pd.read_csv(instancePath, sep=';', header=None).T.to_numpy()
    xs, ys, costs = instance
    
    instanceName = instancePath.split('/')[-1].split('.')[0]
    edges = []
    solutionNames = []
    for edgesPath in glob(os.path.join(solutionsPath, instanceName, "*.csv")):
        # Read generated edges
        tmp1 = pd.read_csv(edgesPath, sep=';', header=None).to_numpy().squeeze()
        tmp2 = np.roll(tmp1, 1)
        edges.append(np.stack([tmp1, tmp2]).T)

        solutionNames.append(edgesPath.split('/')[-1].split('.')[0])
    # Make plots
    fig, axs = plt.subplots(1, len(solutionNames), figsize=(16, 5), dpi = 200)
    fig.suptitle(instanceName, fontsize=20)
    
    for i, ax in enumerate(axs.flat):
        # Plot coords
        ax.scatter(xs, ys, s=costs/10)
        # Plot edges
        for idx1, idx2 in edges[i]:
            ax.plot([xs[idx1], xs[idx2]], [ys[idx1], ys[idx2]], 
                     color=colors[i])
        
        ax.set_title(solutionNames[i])
        ax.set(xlabel='x', ylabel='y')
        ax.label_outer()
    # Save
    fig.savefig(os.path.join(outputPath, instanceName + ".jpg"))

    # # Singe plot
    # fig = plt.figure(figsize=(16, 5), dpi = 200) 
    # fig.suptitle(instanceName, fontsize=20)
    # i = 0
    # # Plot coords
    # plt.scatter(xs, ys, s=costs/10)
    # # Plot edges
    # for idx1, idx2 in edges[i]:
    #     plt.plot([xs[idx1], xs[idx2]], [ys[idx1], ys[idx2]], 
    #              color=colors[i])
    # fig.savefig('./' + instanceName + ".jpg")
