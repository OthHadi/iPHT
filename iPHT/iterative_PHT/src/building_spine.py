#####################################################################
# Copyright (c) 2026 Othman Alghamdi
# email: othhadi@hotmail.com
# All rights reserved.

#####################################################################
import math
import numpy as np
from src.share.Phase3 import *
from src.share.Phase2 import Phase2
import collections.abc

Global_w = np.zeros(3*3).reshape(3,3)

def run(rrr, data_x, label_, Hough_Accumulate_ALL, level, RUN):

    label = label_
    unique_labels = np.unique(label_)
    label_count = 0
    for lbl in unique_labels:
        label[label_ == lbl] = label_count
        label_count += 1

    #####################################################################
    N_c = len(np.unique(label))
    spine_curve = np.zeros(((N_c),12))

    distance_of_nearest_neighbors = rrr
    p2 = Phase2()
    a, b, x_polyhedra, y_polyhedra, z_polyhedra = p2.polyhedra_octahedral_uniform(distance_of_nearest_neighbors,level)
    sum_edge_distance = 0
    
    ######################PCA###########################
    for index_Q in range(1,N_c+1,1):

        print('Filament ',index_Q, ' of ', N_c)
        mask_label = np.where(label == index_Q)[0]
        first_end_Phase2 = -1
        second_end_Phase2 = -1
        C_X = data_x[mask_label,0]
        C_Y = data_x[mask_label,1]
        C_Z = data_x[mask_label,2]
        endx = np.zeros(len(C_X))
        endy = np.zeros(len(C_X))
        endz = np.zeros(len(C_X))

        if(len(C_X) != 0):
            xxx_, yyy_, zzz_, coef = Phase3.polynomial_regression3d(C_X, C_Y, C_Z, 2)
            
            spine_curve[index_Q-1,0] = index_Q
            spine_curve[index_Q-1,1:10] = coef.reshape(1, 9)
            spine_curve[index_Q-1,10] = 0
            spine_curve[index_Q-1,11] = 1
        Hough_Accumulate = Hough_Accumulate_ALL[mask_label,:]

        for c in range(len(C_X)):
            endx[c] = x_polyhedra[np.argmax(Hough_Accumulate[c, :])]+C_X[c]
            endy[c] = y_polyhedra[np.argmax(Hough_Accumulate[c, :])]+C_Y[c]
            endz[c] = z_polyhedra[np.argmax(Hough_Accumulate[c, :])]+C_Z[c]

        #####################################################################
        if (len(C_X)>3):
                pca = p2.this_PCA(C_X, C_Y, C_Z)
                C_X2 = pca[:, 0]
                C_Y2 = pca[:, 1]

                max_index_1 = int(np.argmin(C_X2))
                max_index_2 = int(np.argmax(C_X2))

                first_end_Phase2 = int(max_index_1)
                second_end_Phase2 = int(max_index_2)

###############################################################################################################################################
###############################################################################################################################################

                xq = C_X
                yq = C_Y
                zq = C_Z
                q_Hough_Accumulate = Hough_Accumulate
                ###
                xxx, yyy, zzz, coef = Phase3.polynomial_regression3d(xq, yq, zq, 2)
                ###
                best_coef = coef
                
                ############### Simulated Annealing ####################

                multi_objective_optimization_range = np.linspace(0,1,6)
                multi_objective_optimization  = np.zeros(multi_objective_optimization_range.shape[0])
                loss_d_w_best = np.zeros(multi_objective_optimization_range.shape[0])
                loss_a_w_best = np.zeros(multi_objective_optimization_range.shape[0])
                x_best_MOO = np.zeros(multi_objective_optimization_range.shape[0]*3*3).reshape(multi_objective_optimization_range.shape[0],3,3)
                first_end = first_end_Phase2
                second_end = second_end_Phase2

                if (first_end_Phase2 == -1):
                    continue
                if (second_end_Phase2 == -1):
                    continue
                for MOO in range(multi_objective_optimization_range.shape[0]):
                    print()
                    print('MOO ', (MOO+1), ' of ',len(multi_objective_optimization_range))

                    if(RUN == True):
                        k_max = 150
                        k = 0
                        T = 5
                    else:
                        k_max = 15000
                        k = 0
                        T = 500

                    

                    index = first_end  # [63, 150]

                    ang = Phase3.angles(int(math.sqrt(len(q_Hough_Accumulate[0, :]))))

                    x = coef
                    x_best_MOO[MOO] = x
                    kai = 0.3
                    l = -21

                    x_best = x
                    vectors = Phase3.vectors_of_q(x_polyhedra, y_polyhedra, z_polyhedra)
                    loss_p, loss_d, loss_function, loss_d_w, loss_a_w = Phase3.Curve_Loss_Function2(x, x[0][0], x[0][1],
                                                                                                    x[0][2], x[1][0],
                                                                                                    x[1][1],
                                                                                                    x[1][2],
                                                                                                    x[2][0], x[2][1],
                                                                                                    x[2][2], xq, yq, zq,
                                                                                                    ang,
                                                                                                    vectors,
                                                                                                    Hough_Accumulate,
                                                                                                    xxx, yyy, zzz,
                                                                                                    1,
                                                                                                    0, True,
                                                                                                    kai)  # loss_function = Phase3.Curve_Loss_Function2(x, x[0][0], x[0][1], x[0][2], x[1][0], x[1][1], x[1][2], x[2][0], x[2][1],x[2][2], xq, yq, zq, ang, vectors, Hough_Accumulate, xxx, yyy, zzz)

                    ########################################################################################################################################

                    e_best = loss_function
                    x_best = x

                    ########################################################################################################################################
                    T_array = []

                    T_array.append(T)
                    best_fetted_curve = loss_d * 1.5

                    k = 0
                    count_itr = 0
                    x = coef

                    while k < k_max:
                        print('RUNNING ', round((round(((k)/k_max), 3)*100),0), '%', end='\r')

                        x_new = Phase3.weights(x, k, k_max)
                        loss_, loss_d, e_new, loss_d_w, loss_a_w  = Phase3.Curve_Loss_Function2(x_new, x_new[0][0], x_new[0][1], x_new[0][2],
                                                                         x_new[1][0],
                                                                         x_new[1][1], x_new[1][2], x_new[2][0], x_new[2][1],
                                                                         x_new[2][2], xq, yq, zq, ang, vectors,
                                                                         Hough_Accumulate,
                                                                         xxx, yyy, zzz,
                                                                         multi_objective_optimization_range[MOO],
                                                                         1-multi_objective_optimization_range[MOO], False, kai)

                        prob = Phase3.probability(loss_function, e_new, T)
                        rand = random.uniform(0, 1)

                        if ((prob > rand) and (loss_d < best_fetted_curve)) == True:
                            start_in = Phase3.get_t(x_new, xq[first_end], yq[first_end], zq[first_end])
                            stop_in = Phase3.get_t(x_new, xq[second_end], yq[second_end], zq[second_end])
                            distance, x_plot,y_plot,z_plot = Phase3.curve_limit_distance(x_new, start_in, stop_in)
                            if (sum_edge_distance/1.9 >= distance):
                                count_itr += 1
                                x = x_new
                                loss_function = e_new

                        if count_itr == 200:
                            x = x_best.copy()
                            count_itr = 0

                        if e_new < e_best:
                            start_in = Phase3.get_t(x_new, xq[first_end], yq[first_end], zq[first_end])
                            stop_in = Phase3.get_t(x_new, xq[second_end], yq[second_end], zq[second_end])
                            distance, x_plot, y_plot, z_plot = Phase3.curve_limit_distance(x_new, start_in, stop_in)
                            if (sum_edge_distance / 1.9 >= distance):
                                x_best = x_new
                                e_best = e_new

                                loss_function = e_new

                                x_best_MOO[MOO] = x_new
                                loss_d_w_best[MOO] =  loss_d
                                loss_a_w_best[MOO] =  loss_

                                x = x_new

                                ########################################################################################################################################

                        T = T * 0.9995
                        T_array.append(T)
                        k += 1
                max_loss_a_w_best = max(loss_a_w_best)

                for MOO in range(multi_objective_optimization_range.shape[0]):
                    multi_objective_optimization[MOO] = Phase2.distance_2d([0, 0], [loss_d_w_best[MOO], max_loss_a_w_best-loss_a_w_best[MOO]])

                x_best = x_best_MOO[np.argmin(multi_objective_optimization)]

                if(sum(sum(x_best)) == 0):
                    x_best = best_coef

                index = first_end_Phase2

                start = Phase3.get_t(x_best, xq[index], yq[index], zq[index])
                index = second_end_Phase2

                stop = Phase3.get_t(x_best, xq[index], yq[index], zq[index])
                xx = np.zeros(100)
                yy = xx.copy()
                zz = xx.copy()

                index = 0
                for i in np.linspace(start, stop, len(xx)):
                    if (index < len(xx)):
                        xx[index] = x_best[0][0] + (x_best[0][1] * i) + (x_best[0][2] * (i ** 2))
                        yy[index] = x_best[1][0] + (x_best[1][1] * i) + (x_best[1][2] * (i ** 2))
                        zz[index] = x_best[2][0] + (x_best[2][1] * i) + (x_best[2][2] * (i ** 2))
                    index += 1

                if(isinstance(start, collections.abc.Sequence)):
                    st = start[0]
                else:
                    st = start
                if(isinstance(stop, collections.abc.Sequence)):
                    sp = stop[0]
                else:
                    sp = stop
                
                spine_curve[index_Q-1,0] = index_Q
                spine_curve[index_Q-1,1:10] = x_best.reshape(1, 9)
                spine_curve[index_Q-1,10] = st
                spine_curve[index_Q-1,11] = sp
                
    xx = np.zeros(100)
    yy = xx.copy()
    zz = xx.copy()
    return spine_curve
