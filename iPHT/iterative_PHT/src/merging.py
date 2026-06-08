#####################################################################
# Copyright (c) 2026 Othman Alghamdi
# email: othhadi@hotmail.com
# All rights reserved.

#####################################################################
import math
import numpy as np
import matplotlib.pyplot as plt
from src.share.Phase3 import *
import os
from collections import deque
from src.share.Polyhedra import Polyhedra

view_beta = 25
view_alpha = 25
view_lim = 1 # 9.3
view_lim2 = -1 # 6.2
ets = 1e-15
markersize_value = 1
figsize_value = 11
#####################################################################################
def curve_length(x_best, start, stop, num_points=100):
    distance = 0
    par = np.linspace(start, stop, num_points)
    for i in range(len(par)):
        if (i + 1 < len(par)):
            x = x_best[0] + (x_best[1] * par[i]) + (x_best[2] * (par[i] ** 2))
            y = x_best[3] + (x_best[4] * par[i]) + (x_best[5] * (par[i] ** 2))
            z = x_best[6] + (x_best[7] * par[i]) + (x_best[8] * (par[i] ** 2))

            x2 = x_best[0] + (x_best[1] * par[i + 1]) + (x_best[2] * (par[i + 1] ** 2))
            y2 = x_best[3] + (x_best[4] * par[i + 1]) + (x_best[5] * (par[i + 1] ** 2))
            z2 = x_best[6] + (x_best[7] * par[i + 1]) + (x_best[8] * (par[i + 1] ** 2))
            distance = distance + math.dist([x, y, z], [x2, y2, z2])

    return distance

#####################################################################################
def read_text(path):
    file_handle = open(path, 'r')
    ss = str(file_handle.readline())
    ss = ss.replace('[', '')
    ss = ss.replace(']', '')
    return float(ss)

#####################################################################################
def find_t_on_corve(x_best, t):
    x = x_best[0] + (x_best[1] * t) + (x_best[2] * (t ** 2))
    y = x_best[3] + (x_best[4] * t) + (x_best[5] * (t ** 2))
    z = x_best[6] + (x_best[7] * t) + (x_best[8] * (t ** 2))

    return x, y, z

#####################################################################################
def derivative_curve_equations(x_best, t):
    v1 = np.zeros(3)

    v1[0] = x_best[1] + (x_best[2] * t * 2)
    v1[1] = x_best[4] + (x_best[5] * t * 2)
    v1[2] = x_best[7] + (x_best[8] * t * 2)

    return v1

#####################################################################################
def Angle_Between_Two_Vectors(a, b):
    a_ = math.sqrt((a[0] ** 2) + (a[1] ** 2) + (a[2] ** 2))
    b_ = math.sqrt((b[0] ** 2) + (b[1] ** 2) + (b[2] ** 2))

    a_b = (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])
    cos_a = a_b / (a_ * b_)
    if (cos_a>=1):
        cos_a = int(cos_a)
    angle = math.acos(cos_a)

    if (angle > math.pi / 2):
        angle = math.pi - angle

    return cos_a, angle

#####################################################################################
def plot_spline_V2(spine_curve, radius=1):
    x_best = spine_curve[1:10]
    start = spine_curve[10]
    stop = spine_curve[11]

    distance = curve_length(x_best, start, stop,1000)

    xx = np.zeros(100)
    yy = xx.copy()
    zz = xx.copy()
    index = 0

    for i in np.linspace(start, stop, len(xx)):
        if (index < len(xx)):
            xx[index], yy[index], zz[index] = find_t_on_corve(x_best, i)
        index += 1
    seg = math.ceil(distance/radius)
    xx2 = np.zeros(seg)
    yy2 = np.zeros(seg)
    zz2 = np.zeros(seg)
    t = np.zeros(seg)
    index = 0

    for i in np.linspace(start, stop, seg):
        if (index < len(xx2)):
            xx2[index], yy2[index], zz2[index] = find_t_on_corve(x_best, i)
            t[index] = i
        index += 1
    return xx2, yy2, zz2, t, x_best

#####################################################################################
def find_connected_filaments(spine_curve, radius):
    x, y, z, t, x_best = plot_spline_V2(spine_curve, radius)
    return x, y, z, t, x_best

#####################################################################################
def find_closest_point_on_curve(x_best, P_start, P_stop, x, y, z):

    x_best_ = np.array([[x_best[0], x_best[1], x_best[2]],
                        [x_best[3], x_best[4], x_best[5]],
                        [x_best[6], x_best[7], x_best[8]]
                        ])

    t = Phase3.get_t(x_best_, x, y, z)
    if P_stop < P_start:
        temp = P_start
        P_start = P_stop
        P_stop = temp
    if ((t >= P_start) and (t <= P_stop)) == True:
        return find_t_on_corve(x_best, t)
    else:
        return 1000,1000,1000

#####################################################################################
def tying_filaments(r, xq, yq, zq, Hough_Accumulate, spine_curve):    
    x_best = spine_curve[1:10]
    P_start = spine_curve[10]
    P_stop = spine_curve[11]

    xfilament_points = []
    yfilament_points = []
    zfilament_points = []
    Hough_Accumulate_filament_points = []

    for index in range(len(xq)):
        x, y, z = find_closest_point_on_curve(x_best, P_start, P_stop, xq[index], yq[index], zq[index])
        dist = math.dist([x, y, z],[xq[index], yq[index], zq[index]])
        if dist <= (r):
            xfilament_points.append(xq[index])
            yfilament_points.append(yq[index])
            zfilament_points.append(zq[index])
            Hough_Accumulate_filament_points.append(Hough_Accumulate[index])

    return xfilament_points ,yfilament_points ,zfilament_points, Hough_Accumulate_filament_points

#####################################################################################
def check_if_exist(start, array, value):
    if (len(array) == 0):
        return False
    for ch in range(start, len(array), 1):
        if value == array[ch]:
            return True
    return False

#####################################################################################
def inhomogenous_lists(my_list,len_list,len_depth):
    my_array = np.zeros(len_list*len_depth).reshape(len_list,len_depth)
    my_array_len = np.zeros(len_list)
    for i in range(len_list):
        row = my_list[i]
        row_list = np.array(row)
        my_array_len[i] = len(row_list)
        for j in range(len(row_list)):
            my_array[i,j] = row_list[j]

    return my_array, my_array_len

#####################################################################################
def Vectors_Calculation(x1, y1, z1, x2, y2, z2):
    return [(x1 - x2), (y1 - y2), (z1 - z2)]

#####################################################################################
def parametric_equations(xq, yq, zq, a, b, c, t):
    x = xq + (a * t)
    y = yq + (b * t)
    z = zq + (c * t)

    return x, y, z

#####################################################################################
import os
def create_folder(path):
    filename = path
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("FOOBAR")
        
#####################################################################################
def count_folders_scandir(directory):
  try:
    folder_count = 0
    with os.scandir(directory) as entries:
      for entry in entries:
        if entry.is_dir():
          folder_count += 1
    return folder_count
  except FileNotFoundError:
    #print(f"Error: Directory not found at {directory}")
    return 0
  except PermissionError:
    #print(f"Error: Permission denied to access {directory}")
    return 0

#####################################################################################

def correct_polar_in_python(polar):
    if (polar < (math.pi / 2)):
        polar = (math.pi / 2) - polar
    else:
        polar = (polar - (math.pi / 2)) * (-1)
    return polar

def Geodesic_Distance3(o1, l1, o2, l2, r):
    azimuthal1, polar1 = o1, correct_polar_in_python(l1)
    azimuthal2, polar2 = o2, correct_polar_in_python(l2)

    D_Lo = azimuthal1 - azimuthal2
    D_La = polar1 - polar2

    P = ((math.sin(D_La / 2) ** 2) +
         (math.cos(polar1) * math.cos(polar2) *
          (math.sin(D_Lo / 2) ** 2)))
    Q1 = 2 * math.asin(math.sqrt(P)) * r
    return Q1  # , Q2

def find_shortest_distance(x, y, z, a, b, radius):
    dist = math.inf

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            check_dist = Geodesic_Distance3(a[i], b[i], a[j], b[j], radius)
            if check_dist < dist:
                dist = check_dist
    return dist

def sing(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    elif n == 0:
        return 0

def Spherical_Coordinates_V2(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    azimuthal = math.atan2(y, x)
    polar = math.atan2(math.sqrt(XsqPlusYsq), z)
    return azimuthal, polar  # correct_polar_in_python(polar)

def Cartesian_Coordinates(azimuthal, polar, r):
    x = r * math.cos(azimuthal) * math.sin(polar)
    y = r * math.sin(azimuthal) * math.sin(polar)
    z = r * math.cos(polar)
    return x, y, z

def polyhedra_octahedral_uniform(radius=1, level=6):
    L = math.sqrt(2 * math.pi) / (3 ** (1 / 4))

    p = Polyhedra(L / math.sqrt(2), level)
    pop = p.get_points_of_polyhedra()
    x = []
    y = []
    z = []
    for i in range(len(pop[:, 0])):
        case = 'g'
        if ((pop[i, 0] >= 0) and (pop[i, 1] >= 0) and (pop[i, 2] >= 0)) and (case == 'i1'):
            zc = ((pop[i, 2] * 2) / (L ** 2)) * ((math.sqrt(2) * L) - pop[i, 2])  # in I1  or first octant
            if ((2 * (pop[i, 0] + pop[i, 1])) != 0):
                my_in = (math.pi * pop[i, 1]) / (2 * (pop[i, 0] + pop[i, 1]))  # in I1 or first octant
            else:
                my_in = 0
            if (zc > 1):
                zc = float(1)
            xc = math.sqrt(1 - (zc ** 2)) * math.cos(my_in)
            yc = math.sqrt(1 - (zc ** 2)) * math.sin(my_in)
            ##print('1 = ', (xc ** 2) + (yc ** 2) + (zc ** 2))
            x.append(xc)
            y.append(yc)
            z.append(zc)
        if (case == 'g'):
            zc = ((pop[i, 2] * 2) / (L ** 2)) * ((math.sqrt(2) * L) - abs(pop[i, 2]))  # in general
            if (pop[i, 0] != 0):
                my_sing = pop[i, 1] / pop[i, 0]
            else:
                my_sing = 0
            if ((2 * ((math.sin(my_sing) * pop[i, 0]) + pop[i, 1])) != 0):
                my_in = (math.pi * pop[i, 1]) / (2 * ((sing(my_sing) * pop[i, 0]) + pop[i, 1]))  # in general
            else:
                my_in = 0
            if (zc > 1):
                zc = float(1)
            if (zc < -1):
                zc = float(-1)
            xc = sing(pop[i, 0]) * math.sqrt(1 - (zc ** 2)) * math.cos(my_in)
            yc = sing(pop[i, 1]) * math.sqrt(1 - (zc ** 2)) * math.sin(my_in)
            x.append(xc)
            y.append(yc)
            z.append(zc)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    x_new = []
    y_new = []
    z_new = []

    a = []
    b = []
    max_zc = max(z)

    for i in range(len(x)):
        azimuthal, polar = Spherical_Coordinates_V2(x[i], y[i], z[i])
        xc, yc, zc = Cartesian_Coordinates(azimuthal, polar, radius)
        if (azimuthal < math.pi and azimuthal >= 0) == True:
            if ((x[i] == 0 and y[i] == 0 and z[i] == max_zc) == False):
                a.append(azimuthal)
                b.append(polar)  # (correct_polar_in_python(polar))
                x_new.append(xc)
                y_new.append(yc)
                z_new.append(zc)
    a = np.array(a)
    b = np.array(b)

    return a, b, x_new, y_new, z_new

#####################################################################################
def run(data, rrr, r_d, label_, Hough_Accumulate_ALL_, spine_curve_, label__, Hough_Accumulate_ALL__, spine_curve__):
    fig = plt.figure(figsize=(figsize_value, figsize_value))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((2, 2, 2))
    ax.view_init(view_beta, view_alpha)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    sigma_r = r_d
    radius = rrr
    i_level = 0#i_level
    sigma_sl = math.inf
    while (sigma_sl >= sigma_r):
        i_level += 1
        a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, i_level)
        sigma_sl = find_shortest_distance(x_polyhedra, y_polyhedra, z_polyhedra, a, b, radius)
    radius = rrr
    level2 = i_level-1
    a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, level2)
    level1 = i_level
    a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, level1)

    number_filaments1 = len(np.unique(label_))
    number_filaments2 = len(np.unique(label__))

    data_r = [f'r{i}' for i in range(0, number_filaments1)]
    data_c = [f'c{i}' for i in range(0, number_filaments2)]

    ax.plot3D(data[:,0], data[:,1], data[:,2], 'o', color='gray', alpha=1, markersize=markersize_value)

    connected_filaments = np.zeros(number_filaments1*number_filaments2).reshape(number_filaments1,number_filaments2)
    for i in range(number_filaments1):
        x1, y1, z1, t1, x_best1 = find_connected_filaments(spine_curve_[i], r_d)
        if len(x1) == 0:
            ##print('ERROR 1')
            continue
        for j in range(number_filaments2):
            x2, y2, z2, t2, x_best2 = find_connected_filaments(spine_curve__[j], r_d)
            if len(x2) == 0:
                ##print('ERROR 2')
                continue
            flag = np.zeros(len(x2))
            count = 0
            if_connected = False
            for f1 in range(len(x1)):
                if (if_connected):
                    break
                distance = []
                f1_index = []
                f2_index = []
                for f2 in range(len(x2)):
                    dist = math.dist([x1[f1], y1[f1], z1[f1]],[x2[f2], y2[f2], z2[f2]])
                    if (dist <= r_d):
                        distance.append(dist)
                        f1_index.append(f1)
                        f2_index.append(f2)

                if (len(distance) == 0):
                    ##print('ERROR 3')
                    continue

                #for distance_index in range(len(distance)):
                shortest_dist_index = np.argmin(distance)
                f1 = f1_index[shortest_dist_index]
                f2 = f2_index[shortest_dist_index]

                v1 = derivative_curve_equations(x_best1, t1[f1])
                xv, yv, zv = parametric_equations(x1[f1], y1[f1], z1[f1], v1[0], v1[1], v1[2], 0.05)
                xv_, yv_, zv_ = parametric_equations(x1[f1], y1[f1], z1[f1], v1[0], v1[1], v1[2], -0.05)
                vm1 = Vectors_Calculation(xv, yv, zv, xv_, yv_, zv_)

                v2 = derivative_curve_equations(x_best2, t2[f2])
                xv2, yv2, zv2 = parametric_equations(x2[f2], y2[f2], z2[f2], v2[0], v2[1], v2[2], 0.05)
                xv2_, yv2_, zv2_ = parametric_equations(x2[f2], y2[f2], z2[f2], v2[0], v2[1], v2[2], -0.05)
                vm2 = Vectors_Calculation(xv2, yv2, zv2, xv2_, yv2_, zv2_)

                cos_a, angle = Angle_Between_Two_Vectors(vm1, vm2)
                dist = math.dist([x1[f1], y1[f1], z1[f1]],[x2[f2], y2[f2], z2[f2]])
                if (flag[f2]==0):
                    if(math.degrees(angle) < 30):
                        count += 1
                        flag[f2] = 0

                        if (count >= 2):
                            connected_filaments[i,j] = 1
    plt.close()

    labeled_pred = 0. * np.ones(data.shape[0], dtype=int)
    connected_list = []
    ID_label = 0

    for i in range(len(connected_filaments)):
        for j in range(len(connected_filaments[0])):
            if connected_filaments[i][j] == 1:
                queue = deque()
                queue.append(data_r[i])
                queue.append(data_c[j])
                connected_filaments[i][j] = 2

                ID_label += 1
                connected_values = []

                indices_r_ = np.where(label_ == (i+1))[0]
                labeled_pred[indices_r_] = ID_label
                indices_c_ = np.where(label__ == (j+1))[0]
                labeled_pred[indices_c_] = ID_label

                while queue:
                    data_x = queue.popleft()
                    connected_values.append(data_x)

                    if (data_x in data_r):
                        i_index = data_r.index(data_x)

                        for j_ in range(len(connected_filaments[0])):
                            if connected_filaments[i_index][j_] == 1:
                                queue.append(data_c[j_])
                                connected_filaments[i_index][j_] = 2

                                indices_c = np.where(label__ == (j_+1))[0]
                                labeled_pred[indices_c] = ID_label

                    elif (data_x in data_c):
                        j_index = data_c.index(data_x)

                        for i_ in range(len(connected_filaments)):
                            if connected_filaments[i_][j_index] == 1:
                                queue.append(data_r[i_])
                                connected_filaments[i_][j_index] = 2

                                indices_r = np.where(label_ == (i_+1))[0]
                                labeled_pred[indices_r] = ID_label

                connected_list.append(connected_values)

    return labeled_pred

#main()
##################################################################
