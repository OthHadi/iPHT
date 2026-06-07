import math

#import matplotlib.pyplot as plt
import numpy as np
import os

from src.share.Phase3 import *
from src.share.Phase2 import Phase2
import collections.abc
import glob

Global_w = np.zeros(3*3).reshape(3,3)
#rrr = [20, 17.5, 15, 12.5, 10, 7.5, 5]

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
    #ex = 50

    figsize_value = 10
    pause = 1
    view_lim = 9.3
    view_lim2 = 6.2
    view_lim3 = 1.1

    markersize_value = 5
    view_beta = 55
    view_alpha = 135
    alpha_value = 0.1
    angle_range = math.pi/3.5
    colors = ['red', 'green', 'blue', 'indigo', 'magenta', 'purple', 'violet', 'brown', 'gold', 'black', 'olive',
              'teal', 'orange']

    a, b, x_polyhedra, y_polyhedra, z_polyhedra = p2.polyhedra_octahedral_uniform(distance_of_nearest_neighbors,level)
    sum_edge_distance = 0
    
    ######################PCA###########################
    for index_Q in range(1,N_c+1,1):

        print('Filament ',index_Q, ' of ', N_c)
        mask_label = np.where(label == index_Q)[0]
        first_end_Phase2 = -1
        second_end_Phase2 = -1
        C_X = data_x[mask_label,0] #np.loadtxt(path+str(index_Q)+"/xq.csv", delimiter=",")
        C_Y = data_x[mask_label,1] #np.loadtxt(path+str(index_Q)+"/yq.csv", delimiter=",")
        C_Z = data_x[mask_label,2] #np.loadtxt(path+str(index_Q)+"/zq.csv", delimiter=",")
        endx = np.zeros(len(C_X))
        endy = np.zeros(len(C_X))
        endz = np.zeros(len(C_X))
        #print("C_X=\n ",C_X)
        if(len(C_X) != 0):
            xxx_, yyy_, zzz_, coef = Phase3.polynomial_regression3d(C_X, C_Y, C_Z, 2)
            
            spine_curve[index_Q-1,0] = index_Q
            spine_curve[index_Q-1,1:10] = coef.reshape(1, 9)
            spine_curve[index_Q-1,10] = 0
            spine_curve[index_Q-1,11] = 1
        Hough_Accumulate = Hough_Accumulate_ALL[mask_label,:]
        #a = p2.angles(int(math.sqrt(len(Hough_Accumulate[0, :]))))

        #####################################################################
        ##fig = #plt.figure(figsize=(figsize_value, figsize_value))
        #ax = #plt.axes(projection='3d')
        #ax.set_box_aspect((1, 1, 1))
        #ax.view_init(view_beta, view_alpha)

        ##ax.plot3D(x[:, 0], x[:, 1], x[:, 2], '.', color='gray', markersize=markersize_value, alpha=alpha_value)
        #ax.plot3D(C_X, C_Y, C_Z, 'r.', markersize=markersize_value)

        ##ax.set_xlim(view_lim2, view_lim)
        ##ax.set_ylim(view_lim2, view_lim)
        ##ax.set_zlim(view_lim2, view_lim)
        #max_a, max_b, temperary_Hough_Accumulate = p2.find_a_b(C_X, a, Hough_Accumulate)
        #vectors = p2.vectors_of_q(x_polyhedra, y_polyhedra, z_polyhedra)
        for c in range(len(C_X)):
            ##print(np.argmax(Hough_Accumulate[c, :]))
            ##print(len(x_polyhedra))
            #ax.plot3D(data_x[:,0], data_x[:,1], data_x[:,2], '.', color='gray', alpha =alpha_value)
            #ax.text(C_X[c], C_Y[c], C_Z[c], str(c), fontsize=markersize_value + 4)
            ##ax.annotate(str(c), (C_X[c], C_Y[c], C_Z[c]), fontsize=12, color='k')
            endx[c] = x_polyhedra[np.argmax(Hough_Accumulate[c, :])]+C_X[c]
            endy[c] = y_polyhedra[np.argmax(Hough_Accumulate[c, :])]+C_Y[c]
            endz[c] = z_polyhedra[np.argmax(Hough_Accumulate[c, :])]+C_Z[c]
            #p2.plot_line_3d_V2(C_X[c], C_Y[c], C_Z[c], a[int(max_a[c])], a[int(max_b[c])], 0.1, ax)
            #ax.plot3D([C_X[c],endx[c]],[C_Y[c],endy[c]],[C_Z[c],endz[c]],color='r')
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_zlabel('z')
        ##plt.savefig(save_path + "results/run/"+str(index_Q + 2)+"/DATA.png")
        ##plt.show(block=False)
        ##plt.pause(pause)
        #plt.close()

        #####################################################################
        if (len(C_X)>3):
                pca = p2.this_PCA(C_X, C_Y, C_Z)

                ##fig = #plt.figure(figsize=(figsize_value, figsize_value))
                ##plt.xlim(-2, 2)
                ##plt.ylim(-view_lim3, view_lim3)
                #ax = #plt.gca()
                #plt.grid()
                #ax.set_aspect('equal', adjustable='box')

                C_X2 = pca[:, 0]
                C_Y2 = pca[:, 1]

                max_index_1 = int(np.argmin(C_X2))
                max_index_2 = int(np.argmax(C_X2))

                #plt.plot(C_X2, C_Y2, 'k.', markersize=2)
                #plt.plot(C_X2[max_index_1], C_Y2[max_index_1], 'bo', markersize=10)
                #plt.plot(C_X2[max_index_2], C_Y2[max_index_2], 'bo', markersize=10)
                ##plt.savefig(save_path + "results/run/"+str(index_Q + 2)+"/graph_denoise" + str(index_Q+2) + ".png")
                #plt.close()

                #####################
                #fig = #plt.figure(figsize=(figsize_value, figsize_value))
                #ax = #plt.axes(projection='3d')
                #ax.set_box_aspect((1, 1, 1))
                #ax.view_init(view_beta, view_alpha)

                #ax.plot3D(C_X, C_Y, C_Z, 'k.', markersize=markersize_value)

                ##ax.text(C_X[int(dist_first_end[dist_index])], C_Y[int(dist_first_end[dist_index])], C_Z[int(dist_first_end[dist_index])], str(int(dist_first_end[dist_index])))
                ##ax.text(C_X[int(dist_second_end[dist_index])], C_Y[int(dist_second_end[dist_index])], C_Z[int(dist_second_end[dist_index])], str(int(dist_second_end[dist_index])))

                first_end_Phase2 = int(max_index_1)
                second_end_Phase2 = int(max_index_2)

                ##ax.set_xlim(view_lim2, view_lim)
                ##ax.set_ylim(view_lim2, view_lim)
                ##ax.set_zlim(view_lim2, view_lim)
                #ax.set_xlabel('x')
                #ax.set_ylabel('y')
                #ax.set_zlabel('z')
                #plt.grid()
                ##plt.savefig(save_path + "results//run/"+str(index_Q + 2)+"/graph_3D.png")
                #plt.close()

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
                #filament = return_Q
                xq = C_X
                yq = C_Y
                zq = C_Z
                q_Hough_Accumulate = Hough_Accumulate

                #print(xq.shape)
                # xq = np.loadtxt("C_X_in.csv", delimiter=",")
                # yq = np.loadtxt("C_Y_in.csv", delimiter=",")
                # zq = np.loadtxt("C_Z_in.csv", delimiter=",")
                ###
                xxx, yyy, zzz, coef = Phase3.polynomial_regression3d(xq, yq, zq, 2)
                ###
                best_coef = coef
                view_beta = 55
                view_alpha = 35
                view_lim = 9.3
                view_lim2 = 6.2
                markersize_value = 2
                figsize_value = 11
                
                ############### Simulated Annealing ####################

                multi_objective_optimization_range = np.linspace(0,1,6)
                multi_objective_optimization  = np.zeros(multi_objective_optimization_range.shape[0])
                loss_d_w_best = np.zeros(multi_objective_optimization_range.shape[0])
                loss_a_w_best = np.zeros(multi_objective_optimization_range.shape[0])
                x_best_MOO = np.zeros(multi_objective_optimization_range.shape[0]*3*3).reshape(multi_objective_optimization_range.shape[0],3,3)
                first_end = first_end_Phase2
                second_end = second_end_Phase2
                #loss_function_ini = loss_function
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
                    # vectors = Phase3.vectors_of_q(xq[index], yq[index], zq[index], ang)
                    # q_Hough_Accumulate = Phase3.kk(5, len(xq), len(ang) * len(ang), q_Hough_Accumulate)
                    # max_a, max_b, temperary_Hough_Accumulate = Phase3.find_a_b(xq, ang, q_Hough_Accumulate)

                    # x = coef
                    # loss_f = Curve_Loss_Function(x, x[0][0], x[0][1], x[0][2], x[1][0], x[1][1], x[1][2], x[2][0], x[2][1], x[2][2], xq,yq, zq, ang, max_a, max_b, xxx, yyy, zzz)

                    # xq2, yq2, zq2 = plot_line_3d_V2(xq[index], yq[index], zq[index], ang[int(max_a[index])], ang[int(max_b[index])],0.1)
                    # weight = vector_weight(xq[index], yq[index], zq[index], xq2, yq2, zq2, np.zeros((3, 3)).reshape(3, 3) + 0.1)
                    # x = weights(weight, k, k_max)

                    x = coef
                    x_best_MOO[MOO] = x
                    kai = 0.3
                    l = -21
                    #print('x = ' + str(x))
                    # x_best = np.zeros(k_max * 3 * 3).reshape(k_max, 3, 3)
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
                    '''
                    #fig = #plt.figure(figsize=(figsize_value, figsize_value))
                    #ax = #plt.axes(projection='3d')
                    #ax.set_box_aspect((2, 2, 2))
                    #ax.view_init(view_beta, view_alpha)
                    index = first_end
                    # #ax.plot3D(xq[index], yq[index], zq[index], 'b*', markersize=markersize_value + 10)
                    #ax.plot3D(xq, yq, zq, 'r.', markersize=markersize_value)

                    # Get range for t, calculate the point at 100 points in this range
                    start = Phase3.get_t(x_best, xq[index], yq[index], zq[index])
                    index = 0
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

                    #ax.plot3D(xx, yy, zz, color='g', markersize=markersize_value + 10)
                    ##ax.set_xlim(view_lim2, view_lim)
                    ##ax.set_ylim(view_lim2, view_lim)
                    ##ax.set_zlim(view_lim2, view_lim)
                    #ax.set_title(str(loss_function) + '=' + str(loss_p) + '+' + str(loss_d))
                    #ax.set_xlabel('x')
                    #ax.set_ylabel('y')
                    #ax.set_zlabel('z')
                    #plt.savefig(save_path + "results//run/" + str(index_Q + 2) + "/00.png")
                    # #plt.show(block=False)
                    # #plt.pause(pause)
                    #plt.close()
                    '''
                    ########################################################################################################################################

                    e_best = loss_function
                    e_print = loss_function
                    x_best = x

                    ########################################################################################################################################
                    '''
                    #fig = #plt.figure(figsize=(figsize_value, figsize_value))
                    #ax = #plt.axes(projection='3d')
                    #ax.set_box_aspect((2, 2, 2))
                    #ax.view_init(view_beta, view_alpha)
                    index = first_end
                    # #ax.plot3D(xq[index], yq[index], zq[index], 'b*', markersize=markersize_value + 10)
                    #ax.plot3D(xq, yq, zq, 'r.', markersize=markersize_value)

                    # Get range for t, calculate the point at 100 points in this range
                    start = Phase3.get_t(x_best, xq[index], yq[index], zq[index])
                    index = second_end
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

                    #ax.plot3D(xx, yy, zz, color='g', markersize=markersize_value + 10)
                    ##ax.set_xlim(view_lim2, view_lim)
                    ##ax.set_ylim(view_lim2, view_lim)
                    ##ax.set_zlim(view_lim2, view_lim)
                    #ax.set_title(str(e_best) + '=' + str(loss_p) + '+' + str(loss_d))
                    #ax.set_xlabel('x')
                    #ax.set_ylabel('y')
                    #ax.set_zlabel('z')
                    ##plt.savefig(save_path + "results//run/" + str(index_Q + 2) + "/" + str(k) + ".png")
                    # #plt.show(block=False)
                    # #plt.pause(pause)
                    #plt.close()
                    '''
                    ########################################################################################################################################
                    T_array = []

                    T_array.append(T)
                    best_fetted_curve = loss_d * 1.5

                    k = 0
                    count_itr = 0
                    x = coef
                    #loss_function = loss_function_ini
                    while k < k_max:
                        print('RUNNING ', round((round(((k)/k_max), 3)*100),0), '%', end='\r')

                        #print(k)
                        x_new = Phase3.weights(x, k, k_max)
                        loss_, loss_d, e_new, loss_d_w, loss_a_w  = Phase3.Curve_Loss_Function2(x_new, x_new[0][0], x_new[0][1], x_new[0][2],
                                                                         x_new[1][0],
                                                                         x_new[1][1], x_new[1][2], x_new[2][0], x_new[2][1],
                                                                         x_new[2][2], xq, yq, zq, ang, vectors,
                                                                         Hough_Accumulate,
                                                                         xxx, yyy, zzz,
                                                                         multi_objective_optimization_range[MOO],
                                                                         1-multi_objective_optimization_range[MOO], False, kai)
                        # e_new = Curve_Loss_Function2(x_new, x_new[0][0], x_new[0][1], x_new[0][2], x_new[1][0], x_new[1][1], x_new[1][2], x_new[2][0], x_new[2][1], x_new[2][2], xq, yq, zq, ang, vectors, Hough_Accumulate, xxx, yyy, zzz)
                        prob = Phase3.probability(loss_function, e_new, T)
                        rand = random.uniform(0, 1)
                        Global_w = e_new

                        if ((prob > rand) and (loss_d < best_fetted_curve)) == True:
                            start_in = Phase3.get_t(x_new, xq[first_end], yq[first_end], zq[first_end])
                            stop_in = Phase3.get_t(x_new, xq[second_end], yq[second_end], zq[second_end])
                            distance, x_plot,y_plot,z_plot = Phase3.curve_limit_distance(x_new, start_in, stop_in)
                            if (sum_edge_distance/1.9 >= distance):
                                count_itr += 1
                                x = x_new
                                loss_function = e_new
                                '''
                                ########################################################################################################################################
                                #fig = #plt.figure(figsize=(figsize_value, figsize_value))
                                #ax = #plt.axes(projection='3d')
                                #ax.set_box_aspect((2, 2, 2))
                                #ax.view_init(view_beta, view_alpha)
                                # #ax.plot3D(xq[index], yq[index], zq[index], 'b*', markersize=markersize_value + 10)
                                #ax.plot3D(xq, yq, zq, 'r.', markersize=markersize_value)
                                #ax.plot3D(x_plot, y_plot, z_plot, 'ko', markersize=markersize_value)
                                #ax.plot3D(xq[first_end], yq[first_end], zq[first_end], 'bo', markersize=markersize_value+10)
                                #ax.plot3D(xq[second_end], yq[second_end], zq[second_end], 'bo', markersize=markersize_value+10)
    
                                xx = np.zeros(100)
                                yy = xx.copy()
                                zz = xx.copy()
                                index = 0
                                for i in np.linspace(start_in, stop_in, len(xx)):
                                    if (index < len(xx)):
                                        xx[index] = x[0][0] + (x[0][1] * i) + (x[0][2] * (i ** 2))
                                        yy[index] = x[1][0] + (x[1][1] * i) + (x[1][2] * (i ** 2))
                                        zz[index] = x[2][0] + (x[2][1] * i) + (x[2][2] * (i ** 2))
                                    index += 1
    
                                #ax.plot3D(xx, yy, zz, color='g', markersize=markersize_value + 10)
                                view_lim2 = 2
                                ##ax.set_xlim(view_lim22, view_lim2)
                                ##ax.set_ylim(view_lim22, view_lim2)
                                ##ax.set_zlim(view_lim22, view_lim2)
                                #ax.set_title('e_new= '+str(e_new)+',  '+str(sum_edge_distance/1.9)+' >= '+str(distance))
                                #ax.set_xlabel('x')
                                #ax.set_ylabel('y')
                                #ax.set_zlabel('z')
                                #plt.savefig(save_path + "results//run/" + str(index_Q + 2) + "/t/" + str(k) + ".png")
                                #plt.close()
                                '''
                        #reset weights

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
                                loss_p_best = loss_
                                loss_d_best = loss_d
                                loss_function = e_new

                                x_best_MOO[MOO] = x_new
                                loss_d_w_best[MOO] =  loss_d
                                loss_a_w_best[MOO] =  loss_

                                x = x_new
                                #print("x_best = " + str(x_best))

                                ########################################################################################################################################
                                '''
                                #fig = #plt.figure(figsize=(figsize_value, figsize_value))
                                #ax = #plt.axes(projection='3d')
                                #ax.set_box_aspect((2, 2, 2))
                                #ax.view_init(view_beta, view_alpha)
                                index = first_end
                                ##ax.plot3D(xq[index], yq[index], zq[index], 'b*', markersize=markersize_value + 10)
                                #ax.plot3D(xq, yq, zq, 'r.', markersize=markersize_value)

                                # Get range for t, calculate the point at 100 points in this range
                                start = Phase3.get_t(x_best, xq[index], yq[index], zq[index])
                                index = second_end
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

                                #ax.plot3D(xx, yy, zz, color='g', markersize=markersize_value + 10)
                                ###ax.set_xlim(view_lim2, view_lim)
                                ###ax.set_ylim(view_lim2, view_lim)
                                ###ax.set_zlim(view_lim2, view_lim)
                                #ax.set_title(str(e_best)+'='+str(loss_p_best)+'+'+str(loss_d_best))
                                #ax.set_xlabel('x')
                                #ax.set_ylabel('y')
                                #ax.set_zlabel('z')
                                ##plt.savefig(save_path + "results//run/"+str(index_Q + 2)+"/" +str(MOO)+ "_"+ str(k) + ".png")
                                #plt.close()
                                '''
                                ########################################################################################################################################

                        T = T * 0.9995
                        T_array.append(T)
                        k += 1
                max_loss_a_w_best = max(loss_a_w_best)
                #print(loss_d_w_best)
                #print(loss_a_w_best)
                #print(max_loss_a_w_best)
                #print(max_loss_a_w_best - loss_a_w_best)
                for MOO in range(multi_objective_optimization_range.shape[0]):
                    multi_objective_optimization[MOO] = Phase2.distance_2d([0, 0], [loss_d_w_best[MOO], max_loss_a_w_best-loss_a_w_best[MOO]])
                #print(loss_a_w_best[np.argmin(multi_objective_optimization)])
                #print(np.argmax(multi_objective_optimization))
                #print(multi_objective_optimization)
                #plt.plot(loss_d_w_best, max_loss_a_w_best-loss_a_w_best,'bo')
                ##plt.plot(loss_d_w_best[np.argmin(multi_objective_optimization)], max_loss_a_w_best-loss_a_w_best[np.argmin(multi_objective_optimization)],'ro')
                #plt.xlabel('sum of distances')
                #plt.ylabel('Max of Sums of Probabilities - Sum of Probabilities')
                ##plt.savefig(save_path + "results//run/" + str(index_Q + 2) + "/multi_objective_optimization.png")
                # #plt.show(block=False)
                # #plt.pause(pause)
                #plt.close()

                #plt.plot(loss_d_w_best[np.where(loss_d_w_best != max(loss_d_w_best))], max_loss_a_w_best-loss_a_w_best[np.where(loss_d_w_best != max(loss_d_w_best))],'bo')
                #plt.plot(loss_d_w_best[np.argmin(multi_objective_optimization)], max_loss_a_w_best-loss_a_w_best[np.argmin(multi_objective_optimization)],'ro')
                #plt.xlabel('sum of distances')
                #plt.ylabel('Max of Sums of Probabilities - Sum of Probabilities')
                ##plt.savefig(save_path + "results//run/" + str(index_Q + 2) + "/multi_objective_optimization_0.png")
                # #plt.show(block=False)
                # #plt.pause(pause)
                #plt.close()

                x_best = x_best_MOO[np.argmin(multi_objective_optimization)]
                #print('e_best = ' + str(e_#print))
                #print('e_best = ' + str(e_best))
                #print()
                #print('x_best = ' + str(x_best))
                if(sum(sum(x_best)) == 0):
                    x_best = best_coef
                file = '1'

                #fig = #plt.figure(figsize=(figsize_value, figsize_value))
                #ax = #plt.axes(projection='3d')
                #ax.set_box_aspect((2, 2, 2))
                #ax.view_init(view_beta, view_alpha)
                index = first_end_Phase2
                #print('first_end_Phase2 = ',first_end_Phase2)
                ##ax.plot3D(xq[index], yq[index], zq[index], 'b*', markersize=markersize_value + 10)
                #ax.plot3D(xq, yq, zq, 'r.', markersize=markersize_value)

                # Get range for t, calculate the point at 100 points in this range
                start = Phase3.get_t(x_best, xq[index], yq[index], zq[index])
                index = second_end_Phase2
                #print('second_end_Phase2 = ',second_end_Phase2)
                stop = Phase3.get_t(x_best, xq[index], yq[index], zq[index])
                xx = np.zeros(100)
                yy = xx.copy()
                zz = xx.copy()

                #print('start t = ' + str(start) + ', stop t = ' + str(stop))

                index = 0
                for i in np.linspace(start, stop, len(xx)):
                    if (index < len(xx)):
                        xx[index] = x_best[0][0] + (x_best[0][1] * i) + (x_best[0][2] * (i ** 2))
                        yy[index] = x_best[1][0] + (x_best[1][1] * i) + (x_best[1][2] * (i ** 2))
                        zz[index] = x_best[2][0] + (x_best[2][1] * i) + (x_best[2][2] * (i ** 2))
                    index += 1

                #ax.plot3D(xx, yy, zz, color='g', markersize=markersize_value)
                ###ax.set_xlim(view_lim2, view_lim)
                ###ax.set_ylim(view_lim2, view_lim)
                ###ax.set_zlim(view_lim2, view_lim)
                #ax.set_xlabel('x')
                #ax.set_ylabel('y')
                #ax.set_zlabel('z')
                ##plt.savefig(save_path + "results//run/"+str(index_Q + 2)+"/7.png")
                # #plt.pause(pause)
                ##plt.show(block=False)
                ##plt.pause(pause)
                #plt.close()

                #plt.plot(T_array)
                #plt.ylabel('T')
                ##plt.savefig(save_path + "results//run/"+str(index_Q + 2)+"/T.png")
                ##plt.show(block=False)
                ##plt.pause(pause)
                #plt.close()

                #ss = [start, stop]
                #print("ss = "+str(ss))
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
