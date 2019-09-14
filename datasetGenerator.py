# -*- coding: utf-8 -*-
"""
@author: Nathan Loretan
"""

from copy import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

markerlist = ["o", "v", "8", "s", "p", "*", "h", "+", "x", "D"]
colorlist =  ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']

def display(dataset, cat, isBlocking=False):
    """Display the different cluster in the dataset and highlight the categories.

    [in]dataset:[[x, y, cat], ]
    [in]cat:[c1, c2, ]              the possible categories of the clusters
    [in]isBlocking:bool             if the pyplot must block the execution"""

    # Plot dataset
    figure = plt.figure("Dataset")

    for x in dataset:
        for c in range(len(cat)):
            print(x[2])
            print(cat[c])
            if x[2] == cat[c] and c < len(markerlist):
                plt.plot(x[0], x[1], colorlist[c] + 'o')
            elif c >= len(markerlist):
                plt.plot(x[0], x[1], 'b' + markerlist[c])

    plt.grid(True)
    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.show(block=isBlocking)

def classification(nbrData, cat, disx, disy, filename="dataset",
                   show=True):

    """Simple classification dataset uniformly distributedself. The dataset is
    saved in a pickle filed. The information saved are: Dataset generated,
    Categories used, min and max x values, min and max y values

    [in]  nbrData:int           number of data to generated for the dataset
    [in]  cat:[c1, c2, ]        possible categories
    [in]  disx:[[min, max], ]   min and max x value for each cluster
    [in]  disy:[[min, max], ]   min and max y value for each cluster
    [in]  filename="dataset"    filename uesd to save the dataset
    [in]  show=True             Display a plot of the dataset

    [out] dataset::[[x, y, cat] ,...]"""

    dataset = list()

    nbr_cluster = len(cat)

    # Define point in each clusters
    for i in range(nbr_cluster):

        low_x  = disx[i][0]
        low_y  = disy[i][0]
        high_x = disx[i][1]
        high_y = disy[i][1]

        for y in range(int(nbrData / nbr_cluster)):
            x = np.random.uniform(low=[low_x, low_y],
                                  high=[high_x, high_y],
                                  size=[2]).tolist()
            x.append(cat[i])
            dataset.append(x)

    shuffle(dataset)

    # Display dataset
    if show:
        display(dataset, np.unique(cat), True)

    # Save the dataset in a file using pickle
    info = (dataset,                        # Dataset generated
            np.unique(cat),                 # Categories used
            [np.min(disx), np.max(disx)],   # min and max x values
            [np.min(disy), np.max(disy)])   # min and max y values

    pickle.dump(info, open('dataset/' + filename,'wb'))

    return dataset

if __name__ == "__main__":

    x = [[0.0, 0.5], [0.5, 1.0]]
    y = [[0.5, 1.0], [0.0, 0.5]]
    classification(300, [0, 1], x, y, 'classification2_1', False)

    x = [[0.0, 0.5], [0.5, 1.0]]
    y = [[0.0, 0.5], [0.5, 1.0]]
    classification(300, [0, 1], x, y, 'classification2_2', False)

    x = [[0.0, 0.5], [0.5, 1.0]]
    y = [[0.0, 0.5], [0.0, 0.5]]
    classification(300, [0, 1], x, y, 'classification2_3', False)

    x = [[0.5, 1.0], [0.5, 1.0]]
    y = [[0.0, 0.5], [0.5, 1.0]]
    classification(300, [0, 1], x, y, 'classification2_4', False)

    x = [[0.0, 0.5], [0.5, 1.0], [0.0, 0.5], [0.5, 1.0]]
    y = [[0.0, 0.5], [0.0, 0.5], [0.5, 1.0], [0.5, 1.0]]
    classification(300, [0, 1.0, 1.0, 0], x, y, 'xor_problem', False)

    x = [[0.0, 0.5], [0.5, 1.0], [0.0, 0.5], [0.5, 1.0]]
    y = [[0.0, 0.5], [0.0, 0.5], [0.5, 1.0], [0.5, 1.0]]
    classification(300, [[0,0], [1,0], [0,1], [1,1]], x, y, 'classification4', False)

    x = [[0.0, 1.0],  [0.0,  1.0], [0.0,  0.25], [0.75, 1.0],  [0.25, 0.75]]
    y = [[0.0, 0.25], [0.75, 1.0], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]
    classification(500, [1, 1, 1, 1, 0], x, y, 'kernel', False)
