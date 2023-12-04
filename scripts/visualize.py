import os
from glob import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


labPath = "../lab6/"
width = 16


instancesPath = "../instances/"
solutionsPath = os.path.join(labPath, "solutions/")
outputPath =  os.path.join(labPath, "plots/")

colors = ['red', 'green', 'blue', "cyan", "magenta", "yellow", "orange", "brown"]

for instancePath in glob(os.path.join(instancesPath, "*.csv")):
    # Read raw Instance
    instance = pd.read_csv(instancePath, sep=';', header=None).T.to_numpy()
    xs, ys, costs = instance
    # Read solutions
    instanceName = instancePath.split('/')[-1].split('.')[0]
    edges, solutionNames = [], []
    for solutionPath in sorted(glob(os.path.join(solutionsPath, instanceName, "*.csv"))):
        # Read generated edges
        tmp1 = pd.read_csv(solutionPath, sep=';', header=None).to_numpy().squeeze()
        tmp2 = np.roll(tmp1, -1)
        edges.append(np.stack([tmp1, tmp2]).T)

        solutionNames.append(solutionPath.split('/')[-1].split('.')[0])
    # Make plot
    figsize = (width, 5)
    fig, axs = plt.subplots(1, len(solutionNames), figsize=figsize, dpi = 150)
    
    fig.suptitle(instanceName, fontsize=20)
    for i, ax in enumerate(fig.axes):
        # Plot coords
        ax.scatter(xs, ys, s=costs/10)
        # Plot edges
        for idx1, idx2 in edges[i]:
            ax.plot([xs[idx1], xs[idx2]], 
                    [ys[idx1], ys[idx2]], 
                    color=colors[i])
        
        ax.set_title(solutionNames[i])
        ax.set(xlabel='x', ylabel='y')
        ax.label_outer()    
    # Save plot
    fig.savefig(os.path.join(outputPath, instanceName + ".jpg"))
