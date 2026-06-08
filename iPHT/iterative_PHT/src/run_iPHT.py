#####################################################################
# Copyright (c) 2026 Othman Alghamdi
# email: othhadi@hotmail.com
# All rights reserved.

#####################################################################
from src.iterative_PHT import main
from src.building_spine import run as run_SP
from src.merging import run as run_M
import numpy as np

def run_iPHT(radius, data,  RUN_FAST=False):

    filter_percentage = 0.95
    sigma_r = radius/3
    radius = np.round(radius, 2)

    ####Run PHT####
    label_sl, Hough_Accumulate_ALL_sl, level_sl = main(radius, filter_percentage, data, '', sigma_r, "sigma_sl")
    label_su, Hough_Accumulate_ALL_su, level_su = main(radius, filter_percentage, data, '', sigma_r, "sigma_sl")

    ####Run SP####
    spine_curve_sl = run_SP(radius, data, label_sl, Hough_Accumulate_ALL_sl, level_sl, RUN_FAST)
    spine_curve_su = run_SP(radius, data, label_su, Hough_Accumulate_ALL_su, level_su, RUN_FAST)

    ####Run M####
    labeled_pred = run_M(data, radius, sigma_r, label_sl, Hough_Accumulate_ALL_sl, spine_curve_sl, label_su, Hough_Accumulate_ALL_su, spine_curve_su)

    return labeled_pred