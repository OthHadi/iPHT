import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from src.share.Phase3 import *
from src.share.Phase2 import Phase2
import os
import shutil
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
    ##print(path)
    #if os.path.exists(path + "/x_best.csv") == False:
    #    #print('no x_best')
    #    return [], [], [], [], []

    #x_best = np.loadtxt(path + "/x_best.csv", delimiter=",")
    #start = read_text(path + "/start.txt")
    #stop = read_text(path + "/stop.txt")

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

    #if (show_samples_points):
    #    ax.plot3D(xx, yy, zz, color=color, linewidth=2)
    #if (show_curve):
    #    ax.plot([xx[0],xx[1]], [yy[0],yy[1]], [zz[0],zz[1]], label=name, color=color)

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

    #if (show_curve):
    #    ax.plot3D(xx2, yy2, zz2, 'o',color='black', markersize=4, alpha =0.5)
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
    
    #folder = subfolder + write
    #p = path1 + folder + path2
    #if os.path.exists(p+"/results/run/"+str(filament_ID+4)+ "/x_best.csv") == False:
    #    return [], [], [], []
    #xq = np.loadtxt(p+str(filament_ID+1)+"/xq.csv", delimiter=",")
    #yq = np.loadtxt(p+str(filament_ID+1)+"/yq.csv", delimiter=",")
    #zq = np.loadtxt(p+str(filament_ID+1)+"/zq.csv", delimiter=",")
    #matching_files = glob.glob(p+str(filament_ID+1)+ '/' + 'Hough_Accumulate_q*.csv')
    ##print(matching_files)
    #Hough_Accumulate = np.loadtxt(matching_files[0], delimiter=",")

    #ax.plot3D(xq,yq,zq, 'o', color=color, markersize=markersize_value + 2)
    #x_best = np.loadtxt(p+"/results/run/"+str(filament_ID+4)+"/x_best.csv", delimiter=",")
    #P_start = read_text(p+"/results/run/"+str(filament_ID+4)+ "/start.txt")
    #P_stop = read_text(p+"/results/run/"+str(filament_ID+4)+ "/stop.txt")
    x_best = spine_curve[1:10]
    P_start = spine_curve[10]
    P_stop = spine_curve[11]

    xfilament_points = []
    yfilament_points = []
    zfilament_points = []
    Hough_Accumulate_filament_points = []

    for index in range(len(xq)):
        x, y, z = find_closest_point_on_curve(x_best, P_start, P_stop, xq[index], yq[index], zq[index])
        ##print([x, y, z])
        dist = math.dist([x, y, z],[xq[index], yq[index], zq[index]])
        if dist <= (r):
            #if my_if:
            #    ax.plot3D(xq[index], yq[index], zq[index], 'o', color=color, markersize=markersize_value + 2)
            #ax.plot3D(x, y, z, '*', color='black', markersize=markersize_value + 2)
            #plt.pause(0.1)
            ##print(dist, ' <= ', radius)
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
    # azimuthal1, polar1 = check_boundary(o1, l1)
    # azimuthal2, polar2 = check_boundary(o2, l2)
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
            # check_dist = math.dist([x[i],y[i],z[i]],[x[j],y[j],z[j]])
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
    # radial = math.sqrt((x**2) + (y**2) + (z**2))
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

    # points of polyhedra = pop
    p = Polyhedra(L / math.sqrt(2), level)
    pop = p.get_points_of_polyhedra()
    x = []
    y = []
    z = []
    for i in range(len(pop[:, 0])):
        # if(pop[i,2] > 0) and ((2 * (pop[i,0] + pop[i,1])) != 0):
        # if((pop[i,0] >= 0) and (pop[i,1] >= 0) and (pop[i,2] >= 0)):
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
    ##print('max_zc = ', max_zc)

    ##print("octahedron length = ", len(x))
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

#####################################################################################
def run(data, rrr, r_d, label_, Hough_Accumulate_ALL_, spine_curve_, label__, Hough_Accumulate_ALL__, spine_curve__):
    fig = plt.figure(figsize=(figsize_value, figsize_value))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((2, 2, 2))
    ax.view_init(view_beta, view_alpha)
    #ax.set_xlim(view_lim2, view_lim)
    #ax.set_ylim(view_lim2, view_lim)
    #ax.set_zlim(view_lim2, view_lim)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')

    #x_data = np.loadtxt("/home/q/PycharmProjects/real_data_3d/Data.csv", delimiter=",")
    '''
    path_1 = "/media/q/O/سطح المكتب/سطح المكتب 4/pub/output 20 088/"
    C_X_in = np.loadtxt(path_1 + str(1) + "/C_X_in.csv", delimiter=",")
    C_Y_in = np.loadtxt(path_1 + str(1) + "/C_Y_in.csv", delimiter=",")
    C_Z_in = np.loadtxt(path_1 + str(1) + "/C_Z_in.csv", delimiter=",")

    C_X_out = np.loadtxt(path_1 + str(1) + "/C_X_out.csv", delimiter=",")
    C_Y_out = np.loadtxt(path_1 + str(1) + "/C_Y_out.csv", delimiter=",")
    C_Z_out = np.loadtxt(path_1 + str(1) + "/C_Z_out.csv", delimiter=",")
    '''
    sigma_r = r_d
    radius = rrr
    i_level = 0#i_level
    sigma_sl = math.inf
    while (sigma_sl >= sigma_r):
        i_level += 1
        a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, i_level)
        sigma_su = sigma_sl
        sigma_sl = find_shortest_distance(x_polyhedra, y_polyhedra, z_polyhedra, a, b, radius)
        ##print(sigma_sl)
        ##print("Level = ", i_level)
        #sigma = sigma_sl

    ex = 1
    pause = 1

    #subfolder = '/' + rrr_c #'pub'
    path2 = "/filament/"
    radius = rrr
    #'/home/q/سطح المكتب/pub/output_realdata4/filament'

    write2 = '/sigma_su'#'/sym_data_r02/sym_data_r02_su'#'/output 20 088' #'/output_realdata4'
    ##print("count_folders_scandir(path1+write2+path2))-1",count_folders_scandir(path1+write2+path2)-1)
    #number_filaments2 = count_folders_scandir(path1+write2+path2)-1
    level2 = i_level-1
    a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, level2)
    number_of_angles2 = len(x_polyhedra)
    #number_of_angles2 = 33
    #level2 = 4

    write1 = '/sigma_sl'#'/sym_data_r02/sym_data_r02_sl'#'/output 20 063' #'/output_realdata5'
    ##print("count_folders_scandir(path1+write1+path2))-1",count_folders_scandir(path1+write1+path2)-1)
    #number_filaments1 = count_folders_scandir(path1+write1+path2)-1
    level1 = i_level
    a, b, x_polyhedra, y_polyhedra, z_polyhedra = polyhedra_octahedral_uniform(radius, level1)
    number_of_angles1 = len(x_polyhedra)
    #number_of_angles1 = 51
    #level1 = 5

    #"/media/q/O/سطح المكتب/سطح المكتب 4/pub/real_data_r01/real_data_r01_sl/filament/8"
    #"/media/q/O/سطح المكتب/سطح المكتب 4/pub/real_data_r01/real_data_r01_sl/filament//run/12"
    
    #rrr = 25
    #lelevel = '20000'
    #path1 = '/media/q/O/MOO/'+lelevel+'/r '+str(rrr)+'/r '+str(rrr)+' n 95/results/'
    #path1 = '/media/q/O/MOO/'+lelevel+'/r '+str(rrr)+'/r '+str(rrr)+' n 90/results/output_sl/filament/'
    #path1 = '/media/q/O/MOO/'+lelevel+'/r '+str(rrr)+'/r '+str(rrr)+' n 85/results/output_sl/filament/'
    #path1 = '/media/q/O/MOO/'+lelevel+'/r '+str(rrr)+'/r '+str(rrr)+' n 80/results/output_sl/filament/'
    #path1 = "/media/q/O/MOO/20000/r 20/r 20 n 95/results/"#"/media/q/O/com/100/" #"/media/q/O/سطح المكتب/سطح المكتب 4/"
    
    #create_folder(path1 + "/" + rrr_c + "/m/1.txt")
    #os.remove(path1 + "/" + rrr_c + "/m/1.txt")
    
    path2 = "/filament/"
    #matching_files = glob.glob('input/*.csv')
    #data = np.loadtxt(matching_files[0], delimiter=",")
    
    #label_ = np.loadtxt(path1 + "/"+rrr_c+"_label_"+rrr_c+".csv", delimiter=",")
    #Hough_Accumulate_ALL_ = np.loadtxt(path1 + "/"+rrr_c+"_Hough_Accumulate_ALL_"+rrr_c+".csv", delimiter=",")
    #spine_curve_ = np.loadtxt(path1 + "/"+rrr_c+"_spine_curve_"+rrr_c+".csv", delimiter=",")
    
    #label__ = np.loadtxt(path1 + "/_"+rrr_c+"_label__"+rrr_c+".csv", delimiter=",")
    #Hough_Accumulate_ALL__ = np.loadtxt(path1 + "/_"+rrr_c+"_Hough_Accumulate_ALL__"+rrr_c+".csv", delimiter=",")
    #spine_curve__ = np.loadtxt(path1 + "/_"+rrr_c+"_spine_curve__"+rrr_c+".csv", delimiter=",")

    spine_curve_

    number_filaments1 = len(np.unique(label_))
    number_filaments2 = len(np.unique(label__))

    data_r = [f'r{i}' for i in range(0, number_filaments1)]
    data_c = [f'c{i}' for i in range(0, number_filaments2)]

    ax.plot3D(data[:,0], data[:,1], data[:,2], 'o', color='gray', alpha=1, markersize=markersize_value)

    connected_filaments = np.zeros(number_filaments1*number_filaments2).reshape(number_filaments1,number_filaments2)
    for i in range(number_filaments1):
        #print("it=",i,", len(it)=",number_filaments1)
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

                        #ax.plot3D(x1[f1], y1[f1], z1[f1], 'o', color='k', markersize=markersize_value )
                        #ax.plot3D(x2[f2], y2[f2], z2[f2], 'o', color='k', markersize=markersize_value )
                        #ax.plot3D([xv, x1[f1]], [yv, y1[f1]], [zv, z1[f1]], '--', color='k', markersize=markersize_value, alpha =0.2)
                        #ax.plot3D([xv_, x1[f1]], [yv_, y1[f1]], [zv_, z1[f1]], '--', color='k', markersize=markersize_value, alpha =0.2)
                        #ax.plot3D([xv2, x2[f2]], [yv2, y2[f2]], [zv2, z2[f2]], '--', color='k', markersize=markersize_value, alpha =0.2)
                        #ax.plot3D([xv2_, x2[f2]], [yv2_, y2[f2]], [zv2_, z2[f2]], '--', color='k', markersize=markersize_value, alpha =0.2)
                        if (count >= 2):
                            connected_filaments[i,j] = 1
                        #if_connected = True
                        #break
            ##print('count= ',count)
            ##print('len(x1)= ',len(x1))
            ##print('len(x2)= ',len(x2),'\n')
        ##print('connected_filaments=\n',connected_filaments,'\n')
        #plt.pause(pause)
        #plt.cla()

        #ax.set_box_aspect((2, 2, 2))
        #ax.set_xlim(view_lim2, view_lim)
        #ax.set_ylim(view_lim2, view_lim)
        #ax.set_zlim(view_lim2, view_lim)
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

    '''
    #ax.view_init(view_beta, view_alpha)
    fig = plt.figure(figsize=(figsize_value, figsize_value))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((2, 2, 2))
    ax.view_init(view_beta, view_alpha)
    #ax.set_xlim(view_lim2, view_lim)
    #ax.set_ylim(view_lim2, view_lim)
    #ax.set_zlim(view_lim2, view_lim)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    new_count = 0
    for i in range(2, number_filaments1+1):
        #print("it=",i,", len(it)=",number_filaments1)

        x = []
        y = []
        z = []
        Hough_Accumulate = []
        level_and_ID = []
        add_once = True
        
        mask_label_ = np.where(label_ == i)[0]
        C_X_ = data[mask_label_,0]
        C_Y_ = data[mask_label_,1]
        C_Z_ = data[mask_label_,2]
        Hough_Accumulate_ = Hough_Accumulate_ALL_[mask_label_,:]

        xfilament_points1, yfilament_points1, zfilament_points1, Hough_Accumulate_filament_points1 = tying_filaments(r_d, C_X_, C_Y_, C_Z_, Hough_Accumulate_, spine_curve_[i-1])
        if (len(xfilament_points1) == 0):
            ##print('ERROR 4')
            continue
        for j in range(2, number_filaments2+1):
            if connected_filaments[i-1,j-1] == 1:
                connected_filaments[i-1, j-1] = 2

                mask_label__ = np.where(label__ == j)[0]
                C_X__ = data[mask_label__,0]
                C_Y__ = data[mask_label__,1]
                C_Z__ = data[mask_label__,2]
                Hough_Accumulate__ = Hough_Accumulate_ALL__[mask_label__,:]

                xfilament_points2, yfilament_points2, zfilament_points2, Hough_Accumulate_filament_points2 = tying_filaments(r_d, C_X__, C_Y__, C_Z__, Hough_Accumulate__, spine_curve__[j-1])
                if (len(xfilament_points1) == 0):
                    ##print('ERROR 5')
                    continue
                if add_once:
                    for index in range(len(xfilament_points1)):
                        if (check_if_exist(0, x, xfilament_points1[index])) == False:
                            x.append(xfilament_points1[index])
                            y.append(yfilament_points1[index])
                            z.append(zfilament_points1[index])
                            Hough_Accumulate.append(Hough_Accumulate_filament_points1[index])
                    add_once = False
                for index in range(len(xfilament_points2)):
                    if (check_if_exist(0, x, xfilament_points2[index])) == False:
                        x.append(xfilament_points2[index])
                        y.append(yfilament_points2[index])
                        z.append(zfilament_points2[index])
                        Hough_Accumulate.append(Hough_Accumulate_filament_points2[index])
                for in_j in range(2, number_filaments1+1):
                    if connected_filaments[in_j-1, j-1] == 1:

                        mask_label_2 = np.where(label_ == in_j)[0]
                        C_X_2 = data[mask_label_2,0]
                        C_Y_2 = data[mask_label_2,1]
                        C_Z_2 = data[mask_label_2,2]
                        Hough_Accumulate_2 = Hough_Accumulate_ALL_[mask_label_2,:]

                        xfilament_points3, yfilament_points3, zfilament_points3, Hough_Accumulate_filament_points3 = tying_filaments(r_d, C_X_2, C_Y_2, C_Z_2, Hough_Accumulate_2, spine_curve_[in_j-1])
                        if (len(xfilament_points1) == 0):
                            ##print('ERROR 6')
                            continue
                        connected_filaments[in_j-1, j-1] = 2
                        for index in range(len(xfilament_points3)):
                            if (check_if_exist(0, x, xfilament_points3[index])) == False:
                                x.append(xfilament_points3[index])
                                y.append(yfilament_points3[index])
                                z.append(zfilament_points3[index])
                                Hough_Accumulate.append(Hough_Accumulate_filament_points3[index])
        if add_once==False:
            ##print(path1+subfolder+"/xq"+str(new_count)+".csv")
            np.savetxt(path1+subfolder+"/m/xq"+str(new_count)+".csv", x, delimiter=",")
            np.savetxt(path1+subfolder+"/m/yq"+str(new_count)+".csv", y, delimiter=",")
            np.savetxt(path1+subfolder+"/m/zq"+str(new_count)+".csv", z, delimiter=",")

            if number_of_angles1<number_of_angles2:
                len_depth = number_of_angles2
            else:
                len_depth = number_of_angles1

            Hough_Accumulate_array, Hough_Accumulate_len = inhomogenous_lists(Hough_Accumulate, len(x), len_depth)
            np.savetxt(path1+subfolder+"/m/Hough_Accumulate_q"+str(new_count)+".csv", Hough_Accumulate_array, delimiter=",")
            np.savetxt(path1+subfolder+"/m/Hough_Accumulate_len"+str(new_count)+".csv", Hough_Accumulate_len, delimiter=",")
            new_count += 1

        ax.plot3D(x, y, z, 'o', color='r', markersize=markersize_value + 2)

        #plt.pause(pause)
    #ax.plot3D(x_data[:,0], x_data[:,1], x_data[:,2], '.', color='gray', markersize=2, alpha=0.3)
    #ax.plot3D(C_X_in, C_Y_in, C_Z_in, 'o', color='gray', markersize=markersize_value)
    #ax.plot3D(C_X_out, C_Y_out, C_Z_out, 'o', color='gray', markersize=markersize_value)

    plt.close()
    '''
    
    
    
    '''
    ax.view_init(view_beta, view_alpha)
    fig = plt.figure(figsize=(figsize_value, figsize_value))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((2, 2, 2))
    ax.view_init(view_beta, view_alpha)
    #ax.set_xlim(view_lim2, view_lim)
    #ax.set_ylim(view_lim2, view_lim)
    #ax.set_zlim(view_lim2, view_lim)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')

    for i in range(number_filaments1):
        Test_Not_Exist = True
        for j in range(number_filaments2):
            if connected_filaments[i,j] == 2:
                Test_Not_Exist = False
                break
        if Test_Not_Exist:
            xfilament_points1, yfilament_points1, zfilament_points1, Hough_Accumulate_filament_points1 = tying_filaments(
                r_d, i, path1, path2, write1, subfolder, 'k', True)
            if (len(xfilament_points1) == 0):
                #print('ERROR 7')
                continue
            #np.savetxt(path1+subfolder+"/m/xq"+str(new_count)+".csv", xfilament_points1, delimiter=",")
            #np.savetxt(path1+subfolder+"/m/yq"+str(new_count)+".csv", yfilament_points1, delimiter=",")
            #np.savetxt(path1+subfolder+"/m/zq"+str(new_count)+".csv", zfilament_points1, delimiter=",")
            #Hough_Accumulate_array, Hough_Accumulate_len = inhomogenous_lists(Hough_Accumulate_filament_points1, len(xfilament_points1), number_of_angles1)
            #np.savetxt(path1+subfolder+"/m/Hough_Accumulate_q"+str(new_count)+".csv", Hough_Accumulate_array, delimiter=",")
            #np.savetxt(path1+subfolder+"/m/Hough_Accumulate_len"+str(new_count)+".csv", Hough_Accumulate_len, delimiter=",")
            #new_count += 1
    
    for j in range(number_filaments2):
        Test_Not_Exist = True
        for i in range(number_filaments1):
            if connected_filaments[i,j] == 2:
                Test_Not_Exist = False
                break
        if Test_Not_Exist:
            xfilament_points1, yfilament_points1, zfilament_points1, Hough_Accumulate_filament_points1 = tying_filaments(
                r_d, j, path1, path2, write2, subfolder, 'b',True)
            if (len(xfilament_points1) == 0):
                #print('ERROR 8')
                continue
            #np.savetxt(path1+subfolder+"/m/xq"+str(new_count)+".csv", xfilament_points1, delimiter=",")
            #np.savetxt(path1+subfolder+"/m/yq"+str(new_count)+".csv", yfilament_points1, delimiter=",")
            #np.savetxt(path1+subfolder+"/m/zq"+str(new_count)+".csv", zfilament_points1, delimiter=",")
            #Hough_Accumulate_array, Hough_Accumulate_len = inhomogenous_lists(Hough_Accumulate_filament_points1, len(xfilament_points1), number_of_angles2)
            #np.savetxt(path1+subfolder+"/m/Hough_Accumulate_q"+str(new_count)+".csv", Hough_Accumulate_array, delimiter=",")
            #np.savetxt(path1+subfolder+"/m/Hough_Accumulate_len"+str(new_count)+".csv", Hough_Accumulate_len, delimiter=",")
            #new_count += 1
    #ax.plot3D(x_data[:,0], x_data[:,1], x_data[:,2], '.', color='gray', markersize=2, alpha=0.3)
    #ax.plot3D(C_X_in, C_Y_in, C_Z_in, 'o', color='gray', markersize=markersize_value)
    #ax.plot3D(C_X_out, C_Y_out, C_Z_out, 'o', color='gray', markersize=markersize_value)

    plt.close()

    ##print('connected_filaments=\n', sum(sum(connected_filaments)), '\n')
    labeled_pred = 0. * np.ones(data.shape[0], dtype=int)  # Default to -1 (indicating no match)
    ID_label = 0
    for i in range(0, new_count):
        ID_label += 1
        xq = np.loadtxt(path1 + subfolder + "/m/xq" + str(i) + ".csv", delimiter=",")
        yq = np.loadtxt(path1 + subfolder + "/m/yq" + str(i) + ".csv", delimiter=",")
        zq = np.loadtxt(path1 + subfolder + "/m/zq" + str(i) + ".csv", delimiter=",")
        Hough_Accumulate_q = np.loadtxt(path1 + subfolder + "/m/Hough_Accumulate_q" + str(i) + ".csv", delimiter=",")
        xyz = np.dstack((xq, yq, zq)).squeeze()
        for idx, row in enumerate(xyz):
            matching_rows = np.all(data == row, axis=1)
            matching_indices = np.where(matching_rows)[0]
            labeled_pred[matching_indices] = ID_label

    f_path = os.path.join(path1, rrr_c, "m")
    shutil.rmtree(f_path)
    
    pred_data = data[labeled_pred != 0]
    pred_mask = labeled_pred != 0
    pred__ = labeled_pred[pred_mask]
    n_pred_data = data[labeled_pred == 0]

    labeled_pred = np.round(labeled_pred).astype(int)
    #np.savetxt(path1 + "/"+rrr_c+"_label_merging.csv", labeled_pred, delimiter=",", fmt='%.0f')
    #print('labeled_pred = ',np.unique(labeled_pred))
    #np.savetxt(path1 + "data.csv", data, delimiter=",")
    fig = plt.figure(figsize=(figsize_value, figsize_value))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((2, 2, 2))
    ax.view_init(view_beta, view_alpha)
    #ax.set_xlim(view_lim2, view_lim)
    #ax.set_ylim(view_lim2, view_lim)
    #ax.set_zlim(view_lim2, view_lim)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.scatter(n_pred_data[:, 0], n_pred_data[:, 1], n_pred_data[:, 2], c='gray', marker=".", alpha=0.005)
    ax.scatter(pred_data[:, 0], pred_data[:, 1], pred_data[:, 2], c=pred__.astype(float), marker=".")
    plt.show()
    '''
    return labeled_pred

#main()
##################################################################

'''
def main_s():
    x_data = np.loadtxt("/home/q/PycharmProjects/real_data_3d/Data.csv", delimiter=",")

    ex = 1
    pause = 0.1

    subfolder = 'pub'
    radius = 0.5
    r_d = radius / 3
    #'/home/q/سطح المكتب/pub/output_realdata4/filament'
    write2 = '/output_realdata4'
    number_filaments2 = 21
    number_of_angles2 = 30
    level2 = 4

    write1 = '/output_realdata5'
    number_filaments1 = 38
    number_of_angles1 = 48
    level1 = 5

    path1 = "/home/q/سطح المكتب/"
    path2 = "/filament/"

    fig = plt.figure(figsize=(figsize_value, figsize_value))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((2, 2, 2))
    ax.view_init(view_beta, view_alpha)
    #ax.set_xlim(view_lim2, view_lim)
    #ax.set_ylim(view_lim2, view_lim)
    #ax.set_zlim(view_lim2, view_lim)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.plot3D(x_data[:,0], x_data[:,1], x_data[:,2], '.', color='gray', markersize=2, alpha=0.3)

    xfilament_points = []
    yfilament_points = []
    zfilament_points = []

    if (number_filaments1 > number_filaments2):
        number_filaments = number_filaments1
    else:
        number_filaments = number_filaments2
    for i in range(number_filaments):
        x = []
        y = []
        z = []
        Hough_Accumulate = []
        level_and_ID = []
        add_once = True


        xfilament_points1, yfilament_points1, zfilament_points1, H_ = tying_filaments(r_d, i, path1, path2, write1, subfolder, ax, 'b')
        xfilament_points2, yfilament_points2, zfilament_points2, H_ = tying_filaments(r_d, i, path1, path2, write2, subfolder, ax, 'r')

        if (len(xfilament_points1) != 0):
            for ii in range(len(xfilament_points1)):
                if (check_if_exist(0, xfilament_points, xfilament_points1[ii])) == False:
                    xfilament_points.append(xfilament_points1[ii])
                    yfilament_points.append(yfilament_points1[ii])
                    zfilament_points.append(zfilament_points1[ii])

        if (len(xfilament_points2) != 0):
            for ii in range(len(xfilament_points2)):
                if (check_if_exist(0, xfilament_points, xfilament_points2[ii])) == False:
                    xfilament_points.append(xfilament_points2[ii])
                    yfilament_points.append(yfilament_points2[ii])
                    zfilament_points.append(zfilament_points2[ii])

        ax.plot3D(xfilament_points, yfilament_points, zfilament_points, 'o', color='r', markersize=markersize_value + 2)
        plt.pause(pause)
    plt.close()

    np.savetxt(path1 + subfolder + "/xfilament_points.csv", xfilament_points, delimiter=",")
    np.savetxt(path1 + subfolder + "/yfilament_points.csv", yfilament_points, delimiter=",")
    np.savetxt(path1 + subfolder + "/zfilament_points.csv", zfilament_points, delimiter=",")

#main_s()
'''
