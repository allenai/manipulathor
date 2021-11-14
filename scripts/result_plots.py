import copy

import matplotlib
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

def plot_files(results, text, linestyle=None, color=None, plot_title = ''):


    #TODO use these colors
    #
    if color is  None:
        c1=(68,114,196)
        c3 = (237,125,49)
        c2=(165,165,165)
        c1, c2, c3 = [np.array(x) / 255. for x in [c1, c2, c3]]
    else:
        c1 = c2 = c3 = color
    if linestyle is None:
        linestyle = 'solid'

    #plan 1
    x_list = [x[0] for x in results]
    PU = [x[1] for x in results]
    SR = [x[2] for x in results]
    SRwD = [x[3] for x in results]
    text = f'({text})' if text != '' else ''
    plt.plot(x_list, PU, linestyle=linestyle, label=f'PU{text}',linewidth=3, color=c1)
    plt.plot(x_list, SR,  linestyle=linestyle, label=f'SR{text}',linewidth=3, color=c2)
    plt.plot(x_list, SRwD,  linestyle=linestyle, label=f'SRwD{text}',linewidth=3, color=c3)
    # plt.xlim(right=4.5)
    plt.legend(loc="best", prop={'size': 11},framealpha=0.2)
    # plt.legend(loc="upper right", prop={'size': 11},framealpha=0.2)
    plt.xlabel(plot_title)
    plt.ylabel('Performance')
    plt.grid(True)






# results = [
#     (0, 81.9,60.2,30.8),
#     (0.1,26.7,14.8,7.57 ),
#     (0.2, 17.3,8.92,4.77),
#     (0.4, 13.8,6.49,2.79),
#     (1, 6.22,1.71,1.08)
# ]

#Noise in location
def noise_in_location():
    Ours = [
        (0.005,80.08,57.7,28.6),
        (0.05,80.1,57.2,28.6),
        (0.25,80.9,54.4,26.7),
        (0.5,79.7,48.8,24.7),
        (1,77.4,45.3,22.3)
    ]

    baseline = [
        (0.005,76.2,57.9,29.3),
        (0.05,72.4,52.3,24.3),
        (0.25,35.8,11.9,5.68),
        (0.5,19.5,3.96,1.53),
        (1,10,1.4,0.9)
    ]
    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 12}

    matplotlib.rc('font', **font)
    plot_files(Ours, 'Ours', 'solid', plot_title='Effect of noise in agent\'s location')
    plot_files(baseline,"ArmPointNav", 'dotted', plot_title='Effect of noise in agent\'s location')
    plt.show()
def noise_in_depth():
    MaskDriven=[
        (0,34.2,12.3,7.57),
        (1,20.5,3.5,2.07),
    ]
    LocationAwareMaskDriven=[
        (0,34.8,11.5,7.6),
        (1,25.3,4.05,1.89),
    ]
    PNEmulator=[
        (0,38.7,11.6,4.59),
        (1,36.3,6.13,2.52),
    ]
    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 12}

    matplotlib.rc('font', **font)
    plot_files(MaskDriven, 'MaskDriven', 'solid', color='blue')
    plot_files(LocationAwareMaskDriven,"LocationAwareMaskDriven", 'dotted', color='green')
    plot_files(PNEmulator,"PNEmulator", 'dashed', color='orange')
    plt.show()
def noise_in_mask():
    RetrievalNoise = [
        (0,0,0,0),
        (0.1,57.7,22.4,10.9),
        (0.3,74.7,47.1,23.3),
        (0.6,80,56.4,28.4),
        (0.9,81.4,58.6,31.2),
        (1,81.2,59.6,31),
    ]

    MisDetection = [
        (0,81.2,59.6,31),
        (0.1,57.7,31.2,14.3),
        (0.2,43.1,22.4,10.4),
        (0.3,33.8,12.5,4.95),
        (0.6,22.1,4.59,1.17),
        (0.9,13.7,2.88,1.17),
        (1,0,0,0),
    ]

    MaskIOU = [
        (0,81.2,59.6,31),
        (0.1,82.4,60.2,33.6),
        (0.3,81.5,59.6,31.5),
        (0.6,80.7,58.6,30.6),
        (0.9,74.3,47.7,23.9),
        (1,0,0,0),
    ]

    def reverse(result):
        for i in range(len(result)):
            result[i] = (1 - result[i][0],result[i][1],result[i][2],result[i][3])
        return result

    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 12}

    matplotlib.rc('font', **font)
    # plot_files(RetrievalNoise, '', 'solid', plot_title='Detection Recall')
    # plot_files(MisDetection, '', 'solid', plot_title='Misdetection rate')
    plot_files(reverse(MaskIOU), '', 'solid', plot_title='Accuracy')
    plt.show()

noise_in_mask()