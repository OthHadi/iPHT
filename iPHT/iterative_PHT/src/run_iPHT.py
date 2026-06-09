#####################################################################
# Copyright (c) 2026 Othman Alghamdi
# email: othhadi@hotmail.com
# All rights reserved.

#####################################################################
from src.iterative_PHT import main
from src.iterative_PHT_ import main as main_
from src.building_spine import run as run_SP
from src.merging import run as run_M
import numpy as np

def run_iPHT(radius, data,  RUN_FAST=False, is_synthetic_dataset=False):

    ####################################
    ####################################
    if(is_synthetic_dataset):
        range_o = radius+1
        x_min, x_max = -range_o, range_o
        mask_o = (
            (data[:, 0] >= x_min) & (data[:, 0] <= x_max) &
            (data[:, 1] >= x_min) & (data[:, 1] <= x_max) &
            (data[:, 2] >= x_min) & (data[:, 2] <= x_max)
        )
        data = data[mask_o]
    ####################################
    ####################################


    filter_percentage = 0.95
    sigma_r = radius/3
    radius = np.round(radius, 2)

    ####Run PHT####
    label_sl, Hough_Accumulate_ALL_sl, level_sl = main(radius, filter_percentage, data, '', sigma_r, "sigma_sl")
    label_su, Hough_Accumulate_ALL_su, level_su = main(radius, filter_percentage, data, '', sigma_r, "sigma_su")

    ####Run SP####
    spine_curve_sl = run_SP(radius, data, label_sl, Hough_Accumulate_ALL_sl, level_sl, RUN_FAST)
    spine_curve_su = run_SP(radius, data, label_su, Hough_Accumulate_ALL_su, level_su, RUN_FAST)

    ####Run M####
    labeled_pred = run_M(data, radius, sigma_r, label_sl, Hough_Accumulate_ALL_sl, spine_curve_sl, label_su, Hough_Accumulate_ALL_su, spine_curve_su)

    ####################################
    ####################################
    if(is_synthetic_dataset):
        range_o = 1
        x_min, x_max = -range_o, range_o
        mask_o = (
            (data[:, 0] >= x_min) & (data[:, 0] <= x_max) &
            (data[:, 1] >= x_min) & (data[:, 1] <= x_max) &
            (data[:, 2] >= x_min) & (data[:, 2] <= x_max)
        )
        labeled_pred[~mask_o] = 0
    ####################################
    ####################################

    #Run PHT Once#
    Hough_Accumulate_ALL = np.zeros(data.shape[0] * (Hough_Accumulate_ALL_su.shape[1])).reshape(data.shape[0], Hough_Accumulate_ALL_su.shape[1])
    labeled_pred_id = np.unique(labeled_pred)
    for i in range(1,len(labeled_pred_id)):
        mask_label = np.where(labeled_pred == labeled_pred_id[i])[0]
        data_iPHT = data[mask_label]

        _, Hough_Accumulate_q, level_q = main_(radius, filter_percentage, data_iPHT, "", sigma_r, "sigma_su")
        Hough_Accumulate_ALL[mask_label] = Hough_Accumulate_q
        Hough_Accumulate_ALL[mask_label,0] = labeled_pred_id[i]

    spine = run_SP(radius, data, labeled_pred, Hough_Accumulate_ALL, level_q, RUN_FAST)

    return data, labeled_pred, spine