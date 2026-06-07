from src.iterative_PHT import main
#from iPHT.iterative_PHT_normal import main
from src.building_spine import run as run_SP
from src.merging import run as run_M

import matplotlib.pyplot as plt
import numpy as np
import math

def plot_3d(data, labels_or):
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
    ax.scatter(n_pred_data[:, 0], n_pred_data[:, 1], n_pred_data[:, 2], c='gray', marker=".", alpha=0.005)
    ax.scatter(pred_data[:, 0], pred_data[:, 1], pred_data[:, 2], c=pred__.astype(float), marker=".")
    plt.show()


RUN_FAST = True
noise = 100
data_filepath = 'input/100_n_300_d_1.csv'
radius = 0.2
filter_percentage = 0.95
sigma_r = radius/3
radius = np.round(radius, 2)

data = np.loadtxt(data_filepath, delimiter=",")

range_o = radius+1
x_min, x_max = -range_o, range_o
mask_o = (
    (data[:, 0] >= x_min) & (data[:, 0] <= x_max) &
    (data[:, 1] >= x_min) & (data[:, 1] <= x_max) &
    (data[:, 2] >= x_min) & (data[:, 2] <= x_max)
)

data = data[mask_o]

####Run PHT####
label_sl, Hough_Accumulate_ALL_sl, level_sl = main(radius, filter_percentage, data, '', sigma_r, "sigma_sl")
label_su, Hough_Accumulate_ALL_su, level_su = main(radius, filter_percentage, data, '', sigma_r, "sigma_sl")

####Run SP####
spine_curve_sl = run_SP(radius, data, label_sl, Hough_Accumulate_ALL_sl, level_sl, RUN_FAST)
spine_curve_su = run_SP(radius, data, label_su, Hough_Accumulate_ALL_su, level_su, RUN_FAST)

####Run M####
labeled_pred = run_M(data, radius, sigma_r, label_sl, Hough_Accumulate_ALL_sl, spine_curve_sl, label_su, Hough_Accumulate_ALL_su, spine_curve_su)

np.savetxt('output/label.csv', labeled_pred, delimiter=',', fmt='%.0f')

plot_3d(data, labeled_pred)