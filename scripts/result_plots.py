import copy

import matplotlib.pyplot as plt
import os
import json
import pdb
import ast
import numpy as np


def dist_two_location(p1, p2):
    p1 = p1['position']
    p2 = p2['position']
    return sum([(p1[k] - p2[k]) ** 2 for k in ['x','y','z']]) ** 0.5

def Sort_Tuple(tup):

    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    return(sorted(tup, key = lambda x: x[0]))

def show_plot(frequence, countertops, objects):

    fig, ax = plt.subplots()
    im = ax.imshow(frequence)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(countertops)))
    ax.set_yticks(np.arange(len(objects)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(countertops)
    ax.set_yticklabels(objects)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(objects)):
        for j in range(len(countertops)):
            text = ax.text(j, i, frequence[i, j],
                           ha="center", va="center", color="w")

    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()

def plot_files(results):

    #plan 1
    x_list = [x[0] for x in results]
    PU = [x[1] for x in results]
    SR = [x[2] for x in results]
    SRwD = [x[3] for x in results]
    plt.plot(x_list, PU, label='PU',linewidth=3)
    plt.plot(x_list, SR, label='SR',linewidth=3)
    plt.plot(x_list, SRwD, label='SRwD',linewidth=3)
    # plt.xlim(right=4.5)
    plt.legend(loc="upper right")
    plt.xlabel('Effect of noise in agent\'s location')
    plt.ylabel('Performance')
    plt.grid(True)




    plt.show()
    pdb.set_trace()

results = [
    (0, 81.9,60.2,30.8),
    (0.1,26.7,14.8,7.57 ),
    (0.2, 17.3,8.92,4.77),
    (0.4, 13.8,6.49,2.79),
    (1, 6.22,1.71,1.08)
]

plot_files(results)