import os
from glob import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


similaritiesPath = os.path.join("./similarities/")
outputPath =  os.path.join("./plots/")

for instancePath in glob(os.path.join(similaritiesPath, "*")):
    instanceName = instancePath.split('/')[-1]
    # Read similarities
    similarityNames, objectives, similarities, correlations = [], [], [], []
    for similaritiesPath in sorted(glob(os.path.join(instancePath, "*.csv"))):
        sim = pd.read_csv(similaritiesPath)
        
        similarityName = similaritiesPath.split('/')[-1].split('.')[0]
        correlation = sim['Objective'].corr(sim['Similarity'], method='pearson')
        similarityNames.append(f"{similarityName} corr = {round(correlation, 2)}")
        # Remove similarity of best to best (100%)
        if "best" in similarityNames[-1]:
            sim = sim[sim['Similarity'] != sim['Similarity'].max()]
            
        objectives.append(sim['Objective'].to_numpy())
        similarities.append(sim['Similarity'].to_numpy())
    # Make plot
    figsize = (10, 10)
    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi = 150, sharex=False)
    
    fig.suptitle(instanceName, fontsize=20)
    for i, ax in enumerate(fig.axes):
        sns.histplot(ax=ax, x = objectives[i], y = similarities[i])
        
        ax.set_title(similarityNames[i])
        if i == 0 or i == 2:
            ax.set(ylabel='similarity')
        else:
            ax.set(ylabel=None)
        if i == 2 or i == 3:
            ax.set(xlabel='objective')
    # Save plot
    fig.savefig(os.path.join(outputPath, instanceName + ".jpg"))
