#####################################################################
# Copyright (c) 2026 Othman Alghamdi
# email: othhadi@hotmail.com
# All rights reserved.

#####################################################################
import time
from scipy.spatial import KDTree
import math
import random
import numpy as np
from collections import deque 
from src.share.Polyhedra import Polyhedra
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import networkx as nx
from src.share.Phase3 import Phase3
from numpy.polynomial import Polynomial

#####################################################################
def Geodesic_Distance3(o1, l1, o2, l2, r):
    azimuthal1, polar1 = o1, correct_polar_in_python(l1)
    azimuthal2, polar2 = o2, correct_polar_in_python(l2)

    D_Lo = azimuthal1 - azimuthal2
    D_La = polar1 - polar2

    P = ((math.sin(D_La / 2) ** 2) +
         (math.cos(polar1) * math.cos(polar2) *
          (math.sin(D_Lo / 2) ** 2)))
    Q1 = 2 * math.asin(math.sqrt(P)) * r

    return Q1

#####################################################################
def correct_polar_in_python(polar):
    if (polar < (math.pi / 2)):
        polar = (math.pi / 2) - polar
    else:
        polar = (polar - (math.pi / 2)) * (-1)

    return polar


#####################################################################
def Cartesian_Coordinates(azimuthal, polar, r):
    x = r * math.cos(azimuthal) * math.sin(polar)
    y = r * math.sin(azimuthal) * math.sin(polar)
    z = r * math.cos(polar)

    return x, y, z


#####################################################################
def is_vector_in_array(vector, array):
    for vec in array:
        if np.array_equal(vec, vector):
            return True
    return False


#####################################################################
def Spherical_Coordinates_V2(x, y, z):
    # radial = math.sqrt((x**2) + (y**2) + (z**2))
    XsqPlusYsq = x ** 2 + y ** 2
    azimuthal = math.atan2(y, x)
    polar = math.atan2(math.sqrt(XsqPlusYsq), z)

    return azimuthal, polar  # correct_polar_in_python(polar)


#####################################################################
def vector_weight(xq, yq, zq, xq2, yq2, zq2, dis):
    x = xq + ((xq2 - xq) * dis)
    y = yq + ((yq2 - yq) * dis)
    z = zq + ((zq2 - zq) * dis)

    return x, y, z


#####################################################################
def polyhedra_octahedral(radius, level=4):
    p = Polyhedra(radius, level)
    points_of_polyhedra = p.get_points_of_polyhedra()

    x = points_of_polyhedra[:, 0]
    y = points_of_polyhedra[:, 1]
    z = points_of_polyhedra[:, 2]

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
            # if ((radius in [xc, yc, zc]) == False and (-radius in [xc, yc, zc]) == False) == True:
            if ((xc == 0 and yc == 0 and zc == max_zc) == False):
                a.append(azimuthal)
                b.append(polar)  # (correct_polar_in_python(polar))
                x_new.append(xc)
                y_new.append(yc)
                z_new.append(zc)
    a = np.array(a)
    b = np.array(b)

    return a, b, x_new, y_new, z_new


#####################################################################
def rotate_points(x, y, z, angle, axis):
    theta = np.deg2rad(angle)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    if (axis == 'z'):

        new_x = x * cos_theta - y * sin_theta
        new_y = x * sin_theta + y * cos_theta
        new_z = z
    elif (axis == 'y'):

        new_x = x * cos_theta + z * sin_theta
        new_y = y
        new_z = -x * sin_theta + z * cos_theta

    rotated_points = np.column_stack((new_x, new_y, new_z))

    return rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2]


#####################################################################
def prior_V4(number_of_r, number_of_angles, change_c_x, change_c_y, change_c_z, radius, a, b,
             x_polyhedra, y_polyhedra, z_polyhedra):
    number_of_alpha_angles = number_of_angles

    r = np.linspace(-1, 1, (number_of_r * 2) + 1)

    r_cos_a_sin_b = np.zeros((number_of_alpha_angles, len(r)))
    r_sin_a_sin_b = np.zeros((number_of_alpha_angles, len(r)))
    r_cos_b = np.zeros((number_of_alpha_angles, len(r)))

    for i in range(len(x_polyhedra)):

        r_cos_a_sin_b[i] = r * x_polyhedra[i] + change_c_x
        r_sin_a_sin_b[i] = r * y_polyhedra[i] + change_c_y
        r_cos_b[i] = r * z_polyhedra[i] + change_c_z

    return a, b, r, r_cos_a_sin_b, r_sin_a_sin_b, r_cos_b


#####################################################################
def fibonacci_sphere(samples=1000, radius=1):
    x = np.zeros(samples)
    y = np.zeros(samples)
    z = np.zeros(samples)
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
    for i in range(samples):
        y[i] = 1 - (i / float(samples - 1)) * 1  # y goes from 1 to -1
        radius = math.sqrt(1 - y[i] * y[i])  # radius at y

        y[i] *= radius
        radius *= radius

        theta = phi * i  # golden angle increment
        x[i] = math.cos(theta) * radius
        z[i] = math.sin(theta) * radius

    return x, y, z


#####################################################################
def prior_V3(number_of_r, number_of_angles, change_c_x, change_c_y, change_c_z, radius):
    number_of_alpha_angles = number_of_angles
    number_of_beta_angles = number_of_angles

    r = np.linspace(-1, 1, (number_of_r * 2) + 1)

    x_fibo, y_fibo, z_fibo = fibonacci_sphere(number_of_alpha_angles, radius)

    r_cos_a_sin_b = np.zeros((number_of_alpha_angles, len(r)))
    r_sin_a_sin_b = np.zeros((number_of_alpha_angles, len(r)))
    r_cos_b = np.zeros((number_of_alpha_angles, len(r)))

    a = np.zeros((number_of_alpha_angles))
    b = np.zeros((number_of_beta_angles))

    for i in range(len(x_fibo)):
        r_cos_a_sin_b[i] = r * x_fibo[i] + change_c_x
        r_sin_a_sin_b[i] = r * y_fibo[i] + change_c_y
        r_cos_b[i] = r * z_fibo[i] + change_c_z

    return a, b, r, r_cos_a_sin_b, r_sin_a_sin_b, r_cos_b


#####################################################################
def posterior_V2(c_x, c_y, c_z, r_x, r_y, r_z, len_t, len_a, len_b, len_r, sig_var, prior_a, prior_b, prior_r):
    pos = np.zeros((len_t, len_a, len_r))

    sig_var = sig_var ** 2
    inv_cov = np.linalg.inv(sig_var * np.eye(3))

    before_exp = 1 / (2 * math.pi * (abs(sig_var ** 1.5)))

    for k in range(len_t):
        delta = np.array([c_x[k] - r_x, c_y[k] - r_y, c_z[k] - r_z])
        # exponential = np.exp(-0.5 * np.sum(delta * np.dot(inv_cov, delta), axis=0))
        exponential = np.exp(-0.5 * np.sum(delta * np.matmul(inv_cov, delta), axis=0))
        likelihood = before_exp * exponential
        pos[k] = likelihood * prior_a * prior_b * prior_r

    return pos


#####################################################################
def posterior_v21(c_x, c_y, c_z, r_x, r_y, r_z, len_t, len_a, len_b, len_r, sig, prior_a, prior_b, prior_r):
    pos = np.zeros((len_t, len_a, len_r))

    sig_var = sig ** 2
    cov = sig_var * np.eye(3)  # Identity matrix of shape (3, 3)
    inv_cov = np.linalg.inv(cov)

    before_exp = 1 / (((2 * math.pi) ** (3 / 2)) * (abs(sig_var ** 1.5)))

    C_x = c_x[:, np.newaxis, np.newaxis]  # Shape (len_t, 1, 1)
    C_y = c_y[:, np.newaxis, np.newaxis]  # Shape (len_t, 1, 1)
    C_z = c_z[:, np.newaxis, np.newaxis]  # Shape (len_t, 1, 1)

    R_x = r_x[np.newaxis, :, :]  # Shape (1, len_a, len_r)
    R_y = r_y[np.newaxis, :, :]  # Shape (1, len_a, len_r)
    R_z = r_z[np.newaxis, :, :]  # Shape (1, len_a, len_r)

    radii_x = C_x - R_x  # Shape (len_t, len_a, len_r)
    radii_y = C_y - R_y  # Shape (len_t, len_a, len_r)
    radii_z = C_z - R_z  # Shape (len_t, len_a, len_r)

    radii = np.stack((radii_x, radii_y, radii_z), axis=-1)  # Shape (len_t, len_a, len_r, 3)
    transformed_radii = np.zeros_like(radii)

    for k in range(3):  # The three dimensions (x, y, z)
        transformed_radii[..., k] = np.dot(radii, inv_cov[k, :])  # Shape (len_t, len_a, len_r)

    mahalanobis_dist = np.sum(radii * transformed_radii, axis=-1)  # Shape (len_t, len_a, len_r)

    likelihood = before_exp * np.exp(-0.5 * mahalanobis_dist)

    if len_t == 0:
        return pos
    pos = likelihood * prior_a * prior_r  # Adjust based on your prior needs

    return pos

#####################################################################
def posterior(c_x, c_y, c_z, r_x, r_y, r_z, len_t, len_a, len_b, len_r, sig, prior_a, prior_b, prior_r):
    pos = np.zeros(len_t * len_a * len_r).reshape(len_t, len_a, len_r)

    sig_var = sig ** 2
    cov = sig_var * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    inv_cov = np.linalg.inv(cov)

    before_exp = 1 / (((2 * math.pi) ** (3 / 2)) * (abs(sig_var * sig_var * sig_var) ** 0.5))
    radii = np.zeros(len_r * 3).reshape(len_r, 3)

    for k in range(len_t):
        for i in range(len_a):
            radii[:, 0] = c_x[k] - r_x[i, :]
            radii[:, 1] = c_y[k] - r_y[i, :]
            radii[:, 2] = c_z[k] - r_z[i, :]
            likelihood = before_exp * np.exp((-1 / 2 * radii @ inv_cov @ radii.transpose()))
            pos[k, i, :] = np.diagonal(likelihood) * prior_a * prior_r

    return pos


#####################################################################
def find_nearest_GC_to_the_center(C_X, C_Y, C_Z, change_c_x, change_c_y, change_c_z, radius, c):
    x = np.dstack((C_X, C_Y, C_Z)).squeeze()
    
    tree = KDTree(x)

    center_point = np.array([change_c_x, change_c_y, change_c_z])
    neighbor_indices = tree.query_ball_point(center_point, r=radius)
    neighbor_indices.remove(c)  # Remove the point itself if needed

    c_x = C_X[neighbor_indices]
    c_y = C_Y[neighbor_indices]
    c_z = C_Z[neighbor_indices]
    Index = neighbor_indices

    return c_x, c_y, c_z, Index


#####################################################################
def find_nearest_GC_to_the_center2(C_X, C_Y, C_Z, change_c_x, change_c_y, change_c_z, index_in,
                                   radius, c):
    x = np.dstack((C_X, C_Y, C_Z)).squeeze()
    
    tree = KDTree(x)

    center_point = np.array([change_c_x, change_c_y, change_c_z])
    neighbor_indices = tree.query_ball_point(center_point, r=radius)
    neighbor_indices.remove(c)  # Remove the point itself if needed

    c_x = C_X[neighbor_indices]
    c_y = C_Y[neighbor_indices]
    c_z = C_Z[neighbor_indices]
    Index = index_in[neighbor_indices]

    return c_x, c_y, c_z, Index


#####################################################################
def posterior_s_a(posterior_a_b_r, len_c_x, len_a, len_b, Hough_Accumulate, c_x):
    N_data = posterior_a_b_r.shape[0]
    N_a = Hough_Accumulate.shape[0]
    posterior_a_b = np.zeros((N_data, N_a))

    posterior_a_b = np.sum(posterior_a_b_r[:, :, :], axis=2)
    sum_posterior = np.sum(posterior_a_b, axis=1, keepdims=True)
    posterior_a_b = np.where(sum_posterior != 0, posterior_a_b / sum_posterior, 0)

    if N_data == 0:
        Hough_Accumulate[:] = 0
    else:
        Hough_Accumulate[:] = np.sum(posterior_a_b, axis=0) / N_data

    return posterior_a_b, Hough_Accumulate

def posterior_s_a_old(posterior_a_b_r, len_c_x, len_a, len_b, Hough_Accumulate, c_x):
    posterior_a_b = np.zeros(len_c_x * len_a).reshape(len_c_x, len_a)

    for i in range(len(c_x)):
        for j in range(len_a):
            posterior_a_b[i, j] = sum(posterior_a_b_r[i, j, :])
        sum_posterior = sum(posterior_a_b[i, :])
        for j in range(len_a):
            if (sum_posterior == 0):
                posterior_a_b[i, j] = 0
            else:
                posterior_a_b[i, j] = posterior_a_b[i, j] / sum_posterior

    for i in range(len_a):
        if (len(c_x) == 0):
            Hough_Accumulate[i] = 0
        else:
            Hough_Accumulate[i] = sum(posterior_a_b[:, i]) / len(c_x)
    #print()
    return posterior_a_b, Hough_Accumulate


#####################################################################
def Entropy(Hough_Accumulate, len_a, len_b, ets=1e-15):
    return (-sum([p * math.log2(p + ets) for p in Hough_Accumulate])) / math.log2(len_a)


#####################################################################
def posterior2_V2(c_x, c_y, c_z, r_x, r_y, r_z, len_t, len_a, len_b, len_r, sig_var, prior_a_b, Index, prior_r):
    pos = np.zeros((len_t, len_a, len_r))
    sig_var_sq = sig_var ** 2
    cov_inv = np.linalg.inv(sig_var_sq * np.eye(3))

    before_exp = 1 / (2 * math.pi * abs(sig_var_sq ** 1.5))

    for k in range(len_t):
        dist = np.array([c_x[k] - r_x, c_y[k] - r_y, c_z[k] - r_z])
        dist_sq = dist @ cov_inv @ dist[:, :, np.newaxis]
        likelihood = before_exp * np.exp2(-0.5 * dist_sq)
        pos[k] = likelihood * prior_a_b[Index[k]] * prior_r

    return pos


#####################################################################
def posterior2_v21(c_x, c_y, c_z, r_x, r_y, r_z, len_t, len_a, len_b, len_r, sig, prior_a_b, Index, prior_r):
    pos = np.zeros((len_t, len_a, len_r))

    sig_var = sig ** 2
    cov = sig_var * np.eye(3)  # Identity matrix of shape (3, 3)
    inv_cov = np.linalg.inv(cov)

    before_exp = 1 / (((2 * math.pi) ** (3 / 2)) * (abs(sig_var ** 1.5)))

    C_x = c_x[:, np.newaxis, np.newaxis]  # Shape (len_t, 1, 1)
    C_y = c_y[:, np.newaxis, np.newaxis]  # Shape (len_t, 1, 1)
    C_z = c_z[:, np.newaxis, np.newaxis]  # Shape (len_t, 1, 1)

    R_x = r_x[np.newaxis, :, :]  # Shape (1, len_a, len_r)
    R_y = r_y[np.newaxis, :, :]  # Shape (1, len_a, len_r)
    R_z = r_z[np.newaxis, :, :]  # Shape (1, len_a, len_r)

    radii_x = C_x - R_x  # Shape (len_t, len_a, len_r)
    radii_y = C_y - R_y  # Shape (len_t, len_a, len_r)
    radii_z = C_z - R_z  # Shape (len_t, len_a, len_r)

    radii = np.stack((radii_x, radii_y, radii_z), axis=-1)  # Shape (len_t, len_a, len_r, 3)
    transformed_radii = np.zeros_like(radii)

    for k in range(3):  # The three dimensions (x, y, z)
        transformed_radii[..., k] = np.dot(radii, inv_cov[k, :])  # Shape (len_t, len_a, len_r)

    mahalanobis_dist = np.sum(radii * transformed_radii, axis=-1)  # Shape (len_t, len_a, len_r)

    likelihood = before_exp * np.exp(-0.5 * mahalanobis_dist)

    if len_t == 0:
        return pos
    pos = likelihood * prior_a_b[Index, :, np.newaxis] * prior_r
    return pos

#####################################################################
def posterior2(c_x, c_y, c_z, r_x, r_y, r_z, len_t, len_a, len_b, len_r, sig, prior_a_b, Index, prior_r):
    pos = np.zeros(len_t * len_a * len_r).reshape(len_t, len_a, len_r)

    sig_var = sig ** 2
    cov = sig_var * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    inv_cov = np.linalg.inv(cov)

    before_exp = 1 / (((2 * math.pi) ** (3 / 2)) * ((sig_var * sig_var * sig_var) ** 0.5))
    radii = np.zeros(len_r * 3).reshape(len_r, 3)

    for k in range(len_t):
        for i in range(len_a):
            radii[:, 0] = c_x[k] - r_x[i, :]
            radii[:, 1] = c_y[k] - r_y[i, :]
            radii[:, 2] = c_z[k] - r_z[i, :]
            likelihood = before_exp * np.exp((-1 / 2 * radii @ inv_cov @ radii.transpose()))
            pos[k, i, :] = np.diagonal(likelihood) * prior_a_b[Index[k], i] * prior_r

    return pos


#####################################################################
def filtering(threshold, entropy, C_X, C_Y, C_Z, index_C):
    if len(index_C) == 0:
        index_C = np.zeros(len(C_X))

    C_X_in = []
    C_Y_in = []
    C_Z_in = []
    index_in = []

    C_X_out = []
    C_Y_out = []
    C_Z_out = []
    index_out = []

    for i in range(len(C_X)):
        if ((entropy[i] < threshold) and (entropy[i] != 0) and (index_C[i] == 0)) == True:
            C_X_in.append(C_X[i])
            C_Y_in.append(C_Y[i])
            C_Z_in.append(C_Z[i])
            index_in.append(i)
        else:
            C_X_out.append(C_X[i])
            C_Y_out.append(C_Y[i])
            C_Z_out.append(C_Z[i])
            index_out.append(i)
            if (index_C[i] != 1):
                entropy[i] = 1

    return C_X_in, C_Y_in, C_Z_in, index_in, C_X_out, C_Y_out, C_Z_out, index_out, entropy


#####################################################################
def check_input(message):
    while True:
        try:
            my_input = float(input(message))
        except ValueError:
            #print("Not an number!")
            continue
        else:
            return my_input


#####################################################################
def kk(T, len_T_X_all, len_a, KK, number_of_angles):
    K = np.zeros(len_T_X_all * number_of_angles).reshape(len_T_X_all, number_of_angles)
    KK_entropy = np.zeros(len_T_X_all)

    for tx in range(len_T_X_all):
        B = 1 / T
        K_SUM = np.zeros(len_a)
        for i in range(len_a):
            K_SUM[i] = (KK[tx, i]) ** B
        for i in range(len_a):
            if (sum(K_SUM) != 0):
                K[tx, i] = (KK[tx, i]) ** B / sum(K_SUM)
            else:
                K[tx, i] = 0

        entropy_ = 0
        for i in range(len_a):
            if (K[tx, i] != 0):
                entropy_ = entropy_ - (K[tx, i] * math.log10(K[tx, i]))
            else:
                entropy_ = entropy_ - 0
        KK_entropy[tx] = entropy_ / math.log10(len_a)
    return KK_entropy

#####################################################################
######################find filaments#################################
def find_a_b(C_X, len_a, Hough_Accumulate):
    all_max_b = np.zeros(len_a)
    index_b = np.zeros(len_a)

    max_a = np.zeros(len(C_X))
    max_b = np.zeros(len(C_X))

    for c in range(len(C_X)):
        end_len_a = len_a
        start_len_a = 0
        for i in range(len_a):
            all_max_b[i] = max(Hough_Accumulate[c, start_len_a:end_len_a])  # (temperary_Hough_Accumulate[c, i, :])
            index_b[i] = np.argmax(Hough_Accumulate[c, start_len_a:end_len_a])  # (temperary_Hough_Accumulate[c, i, :])
            start_len_a = end_len_a
            end_len_a = end_len_a + len_a

        max_b[c] = np.argmax(all_max_b)
        max_a[c] = index_b[np.argmax(all_max_b)]

    return max_a, max_b, Hough_Accumulate


#####################################################################
def angles(number_of_angles):
    a = np.zeros((number_of_angles))

    for i in range(number_of_angles):
        a[i] = math.pi / number_of_angles * i

    return a


#####################################################################
def plot_line_3d_V2(x, y, z, a, b, length):
    endx = x + (length * (math.cos(a) * math.sin(b)))
    endy = y + (length * (math.sin(a) * math.sin(b)))
    endz = z + (length * (math.cos(b)))

    return endx, endy, endz


#####################################################################
def check_if_exist(start, array, value):
    for ch in range(start, len(array), 1):
        if value == array[ch]:
            return True
    return False


#####################################################################
def remove_repetitive(array):
    output = []
    output.append(array[0])
    for i in range(1, len(array)):
        if (check_if_exist(0, output, array[i]) == False):
            output.append(array[i])
    return output


#####################################################################
def chech_maching(data, labels, stream, number_of_labels=3):
    my_list = []
    for i in range(len(labels)):
        for j in range(len(stream)):
            if data[i, 0] == stream[j]:
                my_list.append(labels[i])
    list_without_repetitive = remove_repetitive(my_list)

    count = []
    for i in range(len(list_without_repetitive)):
        count.append(my_list.count(list_without_repetitive[i]))
    count = np.array(count)

    max_labels_in_ascending_order = []
    for i in range(number_of_labels):
        max = count.argmax()
        #print(max, ' co=', count[max])
        count[max] = -i
        max_labels_in_ascending_order.append(list_without_repetitive[max])

    return max_labels_in_ascending_order


#####################################################################
def return_index(start, array, value):
    for ch in range(start, len(array), 1):
        if value == array[ch]:
            return ch
    return -1


#####################################################################
def segmentation(len_C_X, graph):
    graph_number = 1
    len_Q = int(len_C_X / 3)
    return_Q = np.ones(len_C_X * len_C_X).reshape(len_C_X, len_C_X)
    return_Q *= -1

    for q in range(len_C_X):
        if (check_if_exist(0, graph[q, :], 1)):
            graph_number = graph_number + 1
            index_Q = graph_number - 2
            Q = []
            Q.append(q)
            for i in range(len_C_X):
                if (graph[q, i] == 1):
                    graph[q, i] = graph_number
                    graph[i, q] = graph_number
                    Q.append(i)
            if (len(Q) > 1):
                z = 0
                break_overloop = 0
                while (True):
                    z += 1
                    break_overloop += 1
                    if (z >= len(Q)):
                        break
                    elif (break_overloop >= 10 * len_C_X):
                        break
                    for j in range(len_C_X):
                        if (graph[Q[z], j] == 1):
                            graph[Q[z], j] = graph_number
                            graph[j, Q[z]] = graph_number
                            if (check_if_exist(0, Q, j) == False):
                                Q.append(j)
            # if (graph_number == 2):
            for i in range(len(Q)):
                return_Q[index_Q, i] = Q[i]

    return graph, return_Q


############################################################################################################
def confusion_matrix(actual_data, predicted_data, noise_data):
    actual_data = np.array(actual_data)
    predicted_data = np.array(predicted_data)
    noise_data = np.array(noise_data)

    tp = 0  # True Positives
    for i in range(actual_data.shape[1]):
        for j in range(predicted_data.shape[1]):
            # if ((actual_data[i, 0] == predicted_data[j, 0] and actual_data[i, 1] == predicted_data[j, 1] and actual_data[i, 2] == predicted_data[j, 2]) == True):
            if ((actual_data[0, i] == predicted_data[0, j] and actual_data[1, i] == predicted_data[1, j] and
                 actual_data[2, i] == predicted_data[2, j]) == True):

                tp += 1
                break
    fn = abs(len(actual_data[1, :]) - tp)  # True Negatives
    fp = abs(len(predicted_data[1, :]) - tp)  # False Positives
    tn = abs(len(noise_data) - fp)  # False Negatives

    return tp, tn, fp, fn


############################################################################################################
def segment_coordinates_nx(coordinates, radius):
    distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2)

    adjacency_matrix = (distances <= radius) & (np.eye(len(coordinates)) == 0)

    g = nx.from_numpy_array(adjacency_matrix.astype(int))

    connected_groups = [g.subgraph(c).copy() for c in nx.connected_components(g)]

    return connected_groups


############################################################################################################
def segment_coordinates_queue(coordinates,distance_of_nearest_neighbors):
    n = len(coordinates)
    visited = set()
    connected_groups = []

    for i in range(n):
        if i not in visited:
            connected_points = []
            bfs(i, coordinates, visited, connected_points,distance_of_nearest_neighbors)
            connected_groups.append(connected_points)

    return connected_groups

def bfs(index, coordinates, visited, connected_points,distance_of_nearest_neighbors):
    visited.add(index)
    queue = deque()
    queue.append(index)

    while queue:
        current_index = queue.popleft()
        connected_points.append(current_index)

        for i, coord in enumerate(coordinates):
            if i not in visited and np.linalg.norm(coord - coordinates[current_index]) <= distance_of_nearest_neighbors:
                visited.add(i)
                queue.append(i)

############################################################################################################
def exponential_func(x, k):
    return np.exp(-1 * k * x)

def find_sim(Hough_Accumulate, data, neighbourhood_size, polyhedra):

    N = data.shape[0]
    k = np.zeros(N)
    for i in range(N):
        neighbourhood_centre = data[i, :]
        neighbourhood_points = np.zeros(N)
        distances = np.linalg.norm(data - neighbourhood_centre, axis=1)
        mask = (distances <= neighbourhood_size)
        neighbourhood_points[mask] = 1
        valid_indices = np.where(neighbourhood_points)[0]
        Hough_Accumulate_q = Hough_Accumulate[valid_indices]
        argmax_indices = np.argmax(Hough_Accumulate_q, axis=1)
        selected_vectors = polyhedra[argmax_indices, :]

        neighbourhood_points_centre = np.zeros(N)
        mask_centre = (distances == 0)
        neighbourhood_points_centre[mask_centre] = 1
        valid_indices_centre = np.where(neighbourhood_points_centre)[0]
        Hough_Accumulate_q_centre = Hough_Accumulate[valid_indices_centre]
        argmax_indices_centre = np.argmax(Hough_Accumulate_q_centre, axis=1)
        selected_vectors_centre = polyhedra[argmax_indices_centre, :]
        selected_vectors_centre = selected_vectors_centre[0,:]

        dot_products = np.dot(selected_vectors, selected_vectors_centre)
        norm_v = np.linalg.norm(selected_vectors_centre)
        norm_v_all = np.linalg.norm(selected_vectors, axis=1)  # Magnitudes of each row in selected_vectors
        cos_angles = dot_products / (norm_v * norm_v_all)
        sim_ = np.clip(cos_angles, -1.0, 1.0)
        sim = np.abs(sim_)

        angles_rad = np.arccos(sim)
        angles_deg = np.degrees(angles_rad)

        accept_sim = np.zeros(Hough_Accumulate_q.shape[0])
        mask_sim = (angles_deg > 35)
        accept_sim[mask_sim] = 1

        k[i] =  sum(accept_sim)

    k_optimal_mean = sum(k)
    return k_optimal_mean

def calculate_distance_3d(point1, point2):
    diff = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2
    return math.sqrt(diff)

def find_k_local_HA(Hough_Accumulate, data, neighbourhood_size):
    N = data.shape[0]
    k = np.zeros(N)
    mean_neighbourhood_number = 0
    # Ensure the vectors are NumPy arrays
    for i in range(N):
        neighbourhood_centre = data[i, :]
        neighbourhood_points = np.zeros(N)

        distances = np.linalg.norm(data - neighbourhood_centre, axis=1)
        mask = (distances <= neighbourhood_size)
        neighbourhood_points[mask] = 1
        valid_indices = np.where(neighbourhood_points)[0]
        Hough_Accumulate_q = Hough_Accumulate[valid_indices]
        mean_neighbourhood_number += Hough_Accumulate_q.shape[0]
        if (Hough_Accumulate_q.shape[0] > 3):
            sim = np.zeros((Hough_Accumulate_q.shape[0], Hough_Accumulate_q.shape[0]))
            for a in range(Hough_Accumulate_q.shape[0]):
                for b in range(Hough_Accumulate_q.shape[0]):
                    sim[a, b] = np.linalg.norm(Hough_Accumulate_q[a, :] - Hough_Accumulate_q[b, :])

            values, vectors = np.linalg.eig(sim)
            values = np.abs(values)  # convert the eigen values to real numbers
            values = np.sort(values)[::-1]  # This sorts the elements in the values array in ascending order.
            original_x = np.arange(len(values))  # contains a sequence of integers from 0 to len(values) - 1.
            rescaled_x = np.linspace(0, len(values) - 1,
                                     100)  # array containing 100 evenly spaced values within the range from 0 to len(values) - 1.
            rescaled_array = interp1d(original_x, values, kind='linear')(rescaled_x)
            popt, pcov = curve_fit(exponential_func, rescaled_x, rescaled_array)
            # Retrieve the optimal value of k
            k[i] =  popt[0]
        else:
            k[i] = 0
            values = np.arange(Hough_Accumulate_q.shape[0])
            sim = np.zeros((Hough_Accumulate_q.shape[0], Hough_Accumulate_q.shape[0]))

    k_optimal_mean = np.mean(k)
    return mean_neighbourhood_number/N, k_optimal_mean, sim, values

def find_k_HA(Hough_Accumulate_q, polyhedra, minmum=4):

    N = Hough_Accumulate_q.shape[0]
    argmax_indices = np.argmax(Hough_Accumulate_q, axis=1)

    sim = np.zeros((Hough_Accumulate_q.shape[0],Hough_Accumulate_q.shape[0]))
    for a in range(Hough_Accumulate_q.shape[0]):
        for b in range(Hough_Accumulate_q.shape[0]):
            sim[a,b] = np.linalg.norm(Hough_Accumulate_q[a,:] - Hough_Accumulate_q[b,:])

    values, vectors = np.linalg.eig(sim)

    values = np.abs(values)  # convert the eigen values to real numbers
    values = values / np.max(values)  # normalize them by divide them on the maximum value
    values = np.sort(values)[::-1]  # This sorts the elements in the values array in ascending order.
    values = values[0:minmum]
    original_x = np.arange(len(values))  # contains a sequence of integers from 0 to len(values) - 1.
    rescaled_x = np.linspace(0, len(values) - 1,
                             100)  # array containing 100 evenly spaced values within the range from 0 to len(values) - 1.
    rescaled_array = interp1d(original_x, values, kind='linear')(rescaled_x)
    popt, pcov = curve_fit(exponential_func, rescaled_x, rescaled_array)
    k_optimal = popt[0]

    return k_optimal, sim, values
############################################################################################################
def find_sim_on_curve(w, a1, b1, c1, a2, b2, c2, a3, b3, c3, xq, yq, zq, vectors, Hough_Accumulate_q,
                     radius, sigma, number_of_r, number_of_angles, x_polyhedra, y_polyhedra, z_polyhedra,
                     a, b):
    angles = np.zeros(len(xq))
    test_angle = 45

    ############# find neighbors for all points #############
    points_c = np.vstack((xq, yq, zq)).T
    tree_c = KDTree(points_c)
    neighbor_lists = tree_c.query_ball_point(points_c, r=radius)
    #########################################################
    Hough_Accumulate_new = np.zeros(len(xq) * number_of_angles).reshape(len(xq), number_of_angles)

    for i, neighbors_c in enumerate(neighbor_lists):
        neighbors_array = np.array(neighbors_c)
        mask_c = neighbors_array != i
        Index = neighbors_array[mask_c]
        c_x = xq[Index]
        c_y = yq[Index]
        c_z = zq[Index]

        a, b, r, r_cos_a_sin_b, r_sin_a_sin_b, r_cos_b = prior_V4(number_of_r, number_of_angles, xq[i], yq[i],
                                                                    zq[i], radius, a, b,
                                                                    x_polyhedra, y_polyhedra, z_polyhedra)

        posterior_a_b_r = posterior_v21(c_x, c_y, c_z, r_cos_a_sin_b, r_sin_a_sin_b, r_cos_b, len(c_x), len(a),
                                    len(b), len(r), sigma, 1 / number_of_angles, 1 / number_of_angles,
                                    1 / len(r))

        posterior_a_b, Hough_Accumulate_new[i] = posterior_s_a(posterior_a_b_r, len(c_x), len(a), len(b),
                                                            Hough_Accumulate_new[i, :], c_x)

        point = [xq[i], yq[i], zq[i]]

        C0, C1, C2, C3, C4 = Phase3.C01234_V2(a1, b1, c1, a2, b2, c2, a3, b3, c3, xq[i], yq[i], zq[i])

        d_sqrd_derivative = Polynomial([C1, C2, C3, C4])
        roots = d_sqrd_derivative.roots()
        three_real = True
        real_index = []

        for j in range(0, len(roots)):
            if np.iscomplex(roots[j]):
                # #print("not real roots = " + str(roots[j]))
                three_real = False
            else:
                # #print("roots = " + str(roots[j]))
                real_index.append(j)

        if three_real:
            root = roots
            closest_point = Phase3.func(w, np.array([[0], [1], [2]]), root)
            t_dists = []
            for k in range(0, len(root)):
                t_dists.append(calculate_distance_3d(point, closest_point[k])) #math.dist(point, closest_point[k]))

            t_dists = np.array(t_dists)
            #distance = t_dists[t_dists.argmin()]
            best_root = root[t_dists.argmin()]

        else:
            best_root = roots[real_index]
        best_root_real = np.real(best_root)
        v1 = Phase3.derivative_curve_equations(b1, c1, b2, c2, b3, c3, best_root_real)
        v2 = vectors[np.argmax(Hough_Accumulate_new[i])]

        dot_product = np.dot(v1, v2)

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        cos_theta_ = dot_product / (norm_v1 * norm_v2)
        cos_theta_ = np.round(cos_theta_, 3)
        cos_theta = abs(cos_theta_)

        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)

        angles[i] = angle_degrees

    accept_sim = np.ones(angles.shape[0])
    mask_sim = (angles < test_angle)
    accept_sim[mask_sim] = 0

    acceptable_stream = len(xq[mask_sim])/len(xq)
    
    if (acceptable_stream >= 0.95):
        return 1, mask_sim
    else:
        return 0, mask_sim

############################################################################################################
def label_matches(x1, x2, label_ID, label):

    dtype = [('col1', float), ('col2', float), ('col3', float)]
    x_view = x1.view(dtype).reshape(-1)
    x1_view = x2.view(dtype).reshape(-1)
    
    matches = np.isin(x1_view, x_view)
    
    label[matches] = label_ID
    
    return label

############################################################################################################

def find_filaments(data_x, x, C_X, C_Y, C_Z, radius, Hough_Accumulate, N_PHT,
                   filament_ID, number_of_angles, number_of_r, x_polyhedra, y_polyhedra, z_polyhedra, a, b,
                   level, name_of_file, which_sigma, label, Hough_Accumulate_ALL, sigma):
    index_C = np.zeros(len(C_X))
    if (True):
        i_path = 1

        coordinates = []
        for i in range(len(C_X)):
            coordinates.append([C_X[i], C_Y[i], C_Z[i]])
        coordinates = np.array(coordinates)
        kdtree = KDTree(coordinates)
        connected_groups = segment_coordinates_nx_fast(coordinates, radius, kdtree)
        a, b, x_polyhedra2, y_polyhedra2, z_polyhedra2 = polyhedra_octahedral_uniform(radius, level)
        polyhedra = np.dstack((x_polyhedra2, y_polyhedra2, z_polyhedra2)).squeeze()
        for fil in connected_groups:
            filament = np.array(fil)
            if (len(filament) >= 4):
                xq = np.zeros(len(filament))
                yq = np.zeros(len(filament))
                zq = np.zeros(len(filament))
                Hough_Accumulate_q = np.zeros(len(filament) * len(Hough_Accumulate[0, :])).reshape(len(filament),
                                                                                                   len(Hough_Accumulate[
                                                                                                       0, :]))
                q_index = np.zeros(len(filament))
                index_c = np.zeros(len(filament))

                for i_filament in range(len(filament)):
                    xq[i_filament] = C_X[int(filament[i_filament])]  # x[where_x, 0]
                    yq[i_filament] = C_Y[int(filament[i_filament])]  # x[where_x, 1]
                    zq[i_filament] = C_Z[int(filament[i_filament])]  # x[where_x, 2]
                    Hough_Accumulate_q[i_filament] = Hough_Accumulate[int(filament[i_filament]), :]
                    q_index[i_filament] = i_filament  # where_x
                    index_c[i_filament] = int(filament[i_filament])  # C_X[int(filament[i_filament])]

                if (True):
                    #####################################################################
                    xxx, yyy, zzz, coef = Phase3.polynomial_regression3d(xq, yq, zq, 2)
                    x = coef
                    enter, mask_sim = find_sim_on_curve(x,
                                            x[0][0], x[0][1], x[0][2],
                                            x[1][0], x[1][1], x[1][2],
                                            x[2][0], x[2][1], x[2][2],
                                            xq, yq, zq, polyhedra, Hough_Accumulate_q,
                                            radius, sigma, number_of_r, number_of_angles,
                                            x_polyhedra, y_polyhedra, z_polyhedra, a, b)

                    if (enter == -1):
                        change_graph = -1
                    elif (enter == 1):
                        filament_ID += 1
                        change_graph = filament_ID
                    if (enter != 0):
                        for i_filament2 in range(len(q_index)):
                            for i_C_X in range(len(C_X)):
                                # if (C_X[i_C_X] == index_c[i_filament2]):
                                if (i_C_X == index_c[i_filament2]):
                                    index_C[i_C_X] = 1
                                    break
                            for j_filament2 in range(len(q_index)):
                                if (enter == 1):
                                    xq_all = np.dstack((xq[mask_sim], yq[mask_sim], zq[mask_sim])).squeeze()
                                    label = label_matches(xq_all, data_x, change_graph, label)
                                    
                                    matching_indices = np.where(label.flatten() == change_graph)[0]
                                    Hough_Accumulate_ALL[matching_indices]=Hough_Accumulate_q[mask_sim]

                                    enter = 0
        i_path += 1

    return index_C, filament_ID, label, Hough_Accumulate_ALL


#####################################################################
#####################################################################
def find_shortest_distance(x, y, z, a, b, radius):
    dist = math.inf

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            # check_dist = math.dist([x[i],y[i],z[i]],[x[j],y[j],z[j]])
            check_dist = Geodesic_Distance3(a[i], b[i], a[j], b[j], radius)
            if check_dist < dist:
                dist = check_dist
    return dist


#####################################################################
def find_number_of_points_on_line(len_samples, radius):
    len_samples = radius / len_samples
    # #print("len_samples = ",len_samples)
    if isinstance(len_samples, int):
        number_of_points = len_samples
    else:
        number_of_points = math.ceil(len_samples)
    return number_of_points


#####################################################################
def sing(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    elif n == 0:
        return 0


#####################################################################
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
            #print('1 = ', (xc ** 2) + (yc ** 2) + (zc ** 2))
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
    #print('max_zc = ', max_zc)

    #print("octahedron length = ", len(x))
    for i in range(len(x)):
        azimuthal, polar = Spherical_Coordinates_V2(x[i], y[i], z[i])
        xc, yc, zc = Cartesian_Coordinates(azimuthal, polar, radius)
        if (azimuthal < math.pi and azimuthal >= 0) == True:
            # if ((radius in [xc, yc, zc]) == False and (-radius in [xc, yc, zc]) == False) == True:
            if ((x[i] == 0 and y[i] == 0 and z[i] == max_zc) == False):
                a.append(azimuthal)
                b.append(polar)  # (correct_polar_in_python(polar))
                x_new.append(xc)
                y_new.append(yc)
                z_new.append(zc)
    a = np.array(a)
    b = np.array(b)

    return a, b, x_new, y_new, z_new

#####################################################################
#####################################################################

def segment_coordinates_nx_fast(coordinates, radius, kdtree):

    if coordinates.size == 0:
        return []

    n_points = coordinates.shape[0]
    connected_groups = []
    visited = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        if visited[i]:
            continue

        group = [i]
        queue = [i]
        visited[i] = True

        while queue:
            current_index = queue.pop(0)
            
            indices_within_radius = kdtree.query_ball_point(coordinates[current_index], radius)
            
            for neighbor_index in indices_within_radius:
                if not visited[neighbor_index]:
                    group.append(neighbor_index)
                    visited[neighbor_index] = True
                    queue.append(neighbor_index)

        connected_groups.append(group)

    return connected_groups

#####################################################################
def main(radius, filter_percentage, x, name_of_file, sigma_r, which_sigma):#, i_level = 4):    
    data_x = np.array(x)
    #####################################################################
    C_X = np.array(x[:, 0])
    C_Y = np.array(x[:, 1])
    C_Z = np.array(x[:, 2])
    
    label = np.zeros(len(C_X))
    ets = 1e-15
    #####################################################################
    # PHT
    ask_me = -1
    N_PHT = 0

    i_level = 0#i_level
    sigma_sl = math.inf
    while (sigma_sl >= sigma_r):
        i_level += 1
        a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, i_level)
        sigma_su = sigma_sl
        sigma_sl = find_shortest_distance(x_polyhedra, y_polyhedra, z_polyhedra, a, b, radius)

    if (which_sigma == "sigma_sl"):
        #sigma = sigma_sl
        level = i_level
        a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, level)
        sigma = find_shortest_distance(x_polyhedra, y_polyhedra, z_polyhedra, a, b, radius)
    if (which_sigma == "sigma_su"):
        #sigma = sigma_su
        level = i_level-1
        a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, level)
        sigma = find_shortest_distance(x_polyhedra, y_polyhedra, z_polyhedra, a, b, radius)
        
    number_of_r = math.ceil(radius / sigma_r)
    number_of_angles = len(x_polyhedra)

    Hough_Accumulate = np.zeros(len(C_X) * number_of_angles).reshape(len(C_X), number_of_angles)
    Hough_Accumulate_ALL = np.zeros(len(C_X) * number_of_angles).reshape(len(C_X), number_of_angles)
    filament_ID = 0

    while (ask_me != 0):
        N_PHT = N_PHT + 1
        count_nies = []

        entropy = np.zeros(len(C_X))
        Hough_Accumulate_old = np.array(Hough_Accumulate)
        Hough_Accumulate = np.zeros(len(C_X) * number_of_angles).reshape(len(C_X), number_of_angles)
        
        percentage_remaining = len(C_X)/len(x[:,0]) * 100
        percentage_str = f"{percentage_remaining:.0f}"

        ############# find neighbors for all points #############
        points_c = np.vstack((C_X, C_Y, C_Z)).T
        tree_c = KDTree(points_c)
        neighbor_lists = tree_c.query_ball_point(points_c, r=radius)
        #########################################################

        for c, neighbors_c in enumerate(neighbor_lists):
            print('radius=', radius, ', ', percentage_str, '%, i=', c, end='\r')
            if (ask_me == -1):
                neighbors_array = np.array(neighbors_c)
                mask_c = neighbors_array != c
                Index = neighbors_array[mask_c]
                c_x = C_X[Index]
                c_y = C_Y[Index]
                c_z = C_Z[Index]
            else:
                neighbors_array = np.array(neighbors_c)
                mask_c = neighbors_array != c
                Index_ = neighbors_array[mask_c]
                c_x = C_X[Index_]
                c_y = C_Y[Index_]
                c_z = C_Z[Index_]
                Index = index_in[Index_]

            a, b, r, r_cos_a_sin_b, r_sin_a_sin_b, r_cos_b = prior_V4(number_of_r, number_of_angles, C_X[c], C_Y[c],
                                                                      C_Z[c], radius, a, b,
                                                                      x_polyhedra, y_polyhedra, z_polyhedra)
            if (ask_me == -1):
                # if (c < 1000):
                if (c < 200):
                    count_nies.append(len(c_x))

                posterior_a_b_r = posterior_v21(c_x, c_y, c_z, r_cos_a_sin_b, r_sin_a_sin_b, r_cos_b, len(c_x), len(a),
                                            len(b), len(r), sigma, 1 / number_of_angles, 1 / number_of_angles,
                                            1 / len(r))
            else:
                count_nies.append(len(c_x))
                posterior_a_b_r = posterior2_v21(c_x, c_y, c_z, r_cos_a_sin_b, r_sin_a_sin_b, r_cos_b, len(c_x), len(a),
                                             len(b), len(r), sigma, Hough_Accumulate_old, Index, 1 / len(r))

            posterior_a_b, Hough_Accumulate[c] = posterior_s_a(posterior_a_b_r, len(c_x), len(a), len(b),
                                                               Hough_Accumulate[c, :], c_x)
            en = Entropy(Hough_Accumulate[c, :], len(a), len(b), ets)
            if (en <= ets):
                entropy[c] = 1
            else:
                entropy[c] = en

        threshold = np.quantile(entropy, filter_percentage)
        index_C, filament_ID, label, Hough_Accumulate_ALL = find_filaments(data_x, x, C_X, C_Y, C_Z, radius,
                                                                 Hough_Accumulate, N_PHT, filament_ID, number_of_angles,
                                                                 number_of_r,
                                                                 x_polyhedra, y_polyhedra, z_polyhedra, a, b,
                                                                 level, name_of_file, which_sigma, label, Hough_Accumulate_ALL,
                                                                 sigma)

        C_X_in, C_Y_in, C_Z_in, index_in_, C_X_out, C_Y_out, C_Z_out, index_out, entropy = filtering(threshold, entropy,
                                                                                                    C_X, C_Y,
                                                                                                    C_Z, index_C)
        #####################################################################

        C_X = np.array(C_X_in)
        C_Y = np.array(C_Y_in)
        C_Z = np.array(C_Z_in)
        index_in = np.array(index_in_)
        #index_out = np.array(index_out_)

        if (len(C_X) <= 4):
            ask_me = 0
        else:
            ask_me = 1

    #####################################################################

    label = np.round(label).astype(int)
    Hough_Accumulate_ALL = np.round(Hough_Accumulate_ALL, 3)

    print('radius = ', radius)
    
    return label, Hough_Accumulate_ALL, level
