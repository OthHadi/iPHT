#####################################################################
# Copyright (c) 2026 Othman Alghamdi
# email: othhadi@hotmail.com
# All rights reserved.

#####################################################################
# wanna use our method? easy:
# from src.run_iPHT import run_iPHT
# _, labeled_pred, spine = run_iPHT(radius, data)
# give it a radius + your (3D) points, get back labels (0 = noise) and the spine curves.

#####################################################################
from src.run_iPHT import run_iPHT
import matplotlib.pyplot as plt
import numpy as np

######################################################################
def plot_curve(ax, spine, value_alpha=1, cc = 'green'):
    for j in range(spine.shape[0]):
        #index_Q = spine[j,0]
        x_best = spine[j,1:10]
        start = spine[j,10]
        stop = spine[j,11]

        xx = np.zeros(100)
        yy = xx.copy()
        zz = xx.copy()
        index = 0
        for i in np.linspace(start, stop, len(xx)):
            xx[index] = x_best[0] + (x_best[1] * i) + (x_best[2] * (i ** 2))
            yy[index] = x_best[3] + (x_best[4] * i) + (x_best[5] * (i ** 2))
            zz[index] = x_best[6] + (x_best[7] * i) + (x_best[8] * (i ** 2))
            index += 1

        ax.plot(xx, yy, zz, color= cc, alpha = value_alpha, linewidth=2, markersize=20)

#####################################################################
def plot_3d(data, labels_or, spine):
    #labels_or = np.loadtxt('1/'+x_input3+'/output/'+str(rri)+'_label_merging.csv', delimiter=",")
    labeled_pred = labels_or#[mask]

    pred_data = data[labeled_pred != 0]
    pred_mask = labeled_pred != 0
    pred__ = labeled_pred[pred_mask]
    n_pred_data = data[labeled_pred == 0]

    #plt.figure(0)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((2, 2, 2))
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plot_curve(ax, spine, value_alpha=1, cc = 'green')
    ax.scatter(n_pred_data[:, 0], n_pred_data[:, 1], n_pred_data[:, 2], c='gray', marker=".", alpha=0.005)
    ax.scatter(pred_data[:, 0], pred_data[:, 1], pred_data[:, 2], c=pred__.astype(float), marker=".")
    plt.show()

######################################################################
RUN_FAST = True
is_synthetic_dataset = True
data_filepath = 'input/5000_n_300_d_1.csv'
radius = 0.15
data = np.loadtxt(data_filepath, delimiter=",")

data_no_boundary, labeled_pred, spine = run_iPHT(radius, data, RUN_FAST=RUN_FAST, is_synthetic_dataset=is_synthetic_dataset)

np.savetxt('output/label.csv', labeled_pred, delimiter=',', fmt='%.0f')
np.savetxt('output/spine.csv', spine, delimiter=',')
if(is_synthetic_dataset):
    np.savetxt('output/data_no_boundary.csv', data_no_boundary, delimiter=',')


plot_3d(data_no_boundary, labeled_pred, spine)

#######################################################################