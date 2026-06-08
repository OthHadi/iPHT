#####################################################################
# Copyright (c) 2026 Othman Alghamdi
# email: othhadi@hotmail.com
# All rights reserved.

#####################################################################
import numpy as np
from sklearn.decomposition import PCA
import math
import os
from src.share.Polyhedra import Polyhedra

class Phase2:
    #####################################################################
    def __init__(self, init=0):
        self.init = init
    #####################################################################
    def Vectors_Calculation(self, x1, y1, x2, y2):
        return [(x1 - x2), (y1 - y2)]

    #####################################################################
    def distance_3d(point1, point2):
        return math.sqrt(
            (point2[0] - point1[0]) ** 2 +
            (point2[1] - point1[1]) ** 2 +
            (point2[2] - point1[2]) ** 2
        )
        
    #####################################################################
    def distance_2d(point1, point2):
        return math.sqrt(
            (point2[0] - point1[0]) ** 2 +
            (point2[1] - point1[1]) ** 2
        )
    
    #####################################################################
    def get_angle_v4(self, x1, y1, x2, y2, C_X2, C_Y2, visited):
        is_unvisited = -1

        if ((x1 <= x2) and (y1 <= y2)) == True:  # 1st quadrant
            if (self.check_if_visited(C_X2, C_Y2, x2, y2, visited) == 1):
                is_unvisited = 0
            q = 0
        elif ((x1 >= x2) and (y1 <= y2)) == True:  # 2nd quadrant
            if (self.check_if_visited(C_X2, C_Y2, x2, y2, visited) == 1):
                is_unvisited = 1
            q = 1
        elif ((x1 >= x2) and (y1 >= y2)) == True:  # 3rd quadrant
            if (self.check_if_visited(C_X2, C_Y2, x2, y2, visited) == 1):
                is_unvisited = 2
            q = 2
        elif ((x1 <= x2) and (y1 >= y2)) == True:  # 4th quadrant
            if (self.check_if_visited(C_X2, C_Y2, x2, y2, visited) == 1):
                is_unvisited = 3
            q = 3

        angle = (q * math.pi / 2) + math.atan2((abs(y2 - y1) / abs(x2 - x1)) ** ((-1) ** q), 1)

        if (angle > math.pi):
            angle = angle - (2 * math.pi)

        '''
        if(abs(angle) > math.pi/2):
            angle = math.pi
        '''

        return math.degrees(angle), q + 1, is_unvisited + 1

    #####################################################################
    def vector_weight(self, xq, yq, zq, xq2, yq2, zq2, dis):
        x = xq + ((xq2 - xq) * dis)
        y = yq + ((yq2 - yq) * dis)
        z = zq + ((zq2 - zq) * dis)

        return x, y, z

    #####################################################################
    def is_vector_in_array(self, vector, array):
        for vec in array:
            if np.array_equal(vec, vector):
                return True
        return False

    #####################################################################
    def Spherical_Coordinates_V2(self, x, y, z):
        # radial = math.sqrt((x**2) + (y**2) + (z**2))
        XsqPlusYsq = x ** 2 + y ** 2
        azimuthal = math.atan2(y, x)
        polar = math.atan2(math.sqrt(XsqPlusYsq), z)

        return azimuthal, polar  # correct_polar_in_python(polar)

    #####################################################################
    def Cartesian_Coordinates(self, azimuthal, polar, r):
        x = r * math.cos(azimuthal) * math.sin(polar)
        y = r * math.sin(azimuthal) * math.sin(polar)
        z = r * math.cos(polar)

        return x, y, z

    #####################################################################
    def Angle_Between_Two_Vectors(self, a, b):
        a_ = math.sqrt((a[0] ** 2) + (a[1] ** 2))
        b_ = math.sqrt((b[0] ** 2) + (b[1] ** 2))

        a_b = (a[0] * b[0]) + (a[1] * b[1])
        cos_a = a_b / (a_ * b_)
        # angle = math.acos(cos_a)

        # if (angle > math.pi / 2):
        # angle = math.pi - angle

        return a_, cos_a

    #####################################################################
    def polyhedra_octahedral_old(self, radius, num=2, plot_me=0):
        hypotenuse = radius * math.sqrt(2)

        x = [radius, -radius, 0, 0, 0, 0]
        y = [0, 0, radius, -radius, 0, 0]
        z = [0, 0, 0, 0, radius, -radius]

        for times in range(num):
            x_new = []
            y_new = []
            z_new = []
            hypo_from = hypotenuse - (hypotenuse / 1000)
            hypo_to = hypotenuse + (hypotenuse / 1000)
            for i in range(len(x)):
                x_new.append(x[i])
                y_new.append(y[i])
                z_new.append(z[i])
                for j in range(i, len(x), 1):
                    distance = Phase2.distance_3d([x[i], y[i], z[i]], [x[j], y[j], z[j]])
                    # print("hypotenuse=",hypotenuse,", distance=",distance)
                    if (distance > hypo_from and distance < hypo_to) == True:
                        xc, yc, zc = self.vector_weight(x[i], y[i], z[i], x[j], y[j], z[j], 0.5)
                        x_new.append(xc)
                        y_new.append(yc)
                        z_new.append(zc)
            x = np.array(x_new)
            y = np.array(y_new)
            z = np.array(z_new)
            hypotenuse = hypotenuse / 2

        x_new = []
        y_new = []
        z_new = []

        a = []
        b = []
        for i in range(len(x)):
            azimuthal, polar = self.Spherical_Coordinates_V2(x[i], y[i], z[i])
            xc, yc, zc = self.Cartesian_Coordinates(azimuthal, polar, radius)
            if (azimuthal < math.pi and azimuthal >= 0) == True:
                if ((radius in [xc, yc, zc]) == False and (-radius in [xc, yc, zc]) == False) == True:
                    a.append(azimuthal)
                    b.append(polar)  # (correct_polar_in_python(polar))
                    x_new.append(xc)
                    y_new.append(yc)
                    z_new.append(zc)
        a = np.array(a)
        b = np.array(b)

        return a, b, x_new, y_new, z_new

    #####################################################################
    def sing(self, n):
        if n > 0:
            return 1
        elif n < 0:
            return -1
        elif n == 0:
            return 0

    #####################################################################
    def polyhedra_octahedral_uniform(self, radius=1, level=6):
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
                    my_in = (math.pi * pop[i, 1]) / (2 * ((self.sing(my_sing) * pop[i, 0]) + pop[i, 1]))  # in general
                else:
                    my_in = 0
                if (zc > 1):
                    zc = float(1)
                if (zc < -1):
                    zc = float(-1)
                xc = self.sing(pop[i, 0]) * math.sqrt(1 - (zc ** 2)) * math.cos(my_in)
                yc = self.sing(pop[i, 1]) * math.sqrt(1 - (zc ** 2)) * math.sin(my_in)
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
            azimuthal, polar = self.Spherical_Coordinates_V2(x[i], y[i], z[i])
            xc, yc, zc = self.Cartesian_Coordinates(azimuthal, polar, radius)
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
    def polyhedra_octahedral(self, radius, level=2, plot_me=0):

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
        for i in range(len(x)):
            azimuthal, polar = self.Spherical_Coordinates_V2(x[i], y[i], z[i])
            xc, yc, zc = self.Cartesian_Coordinates(azimuthal, polar, radius)
            if (azimuthal < math.pi and azimuthal >= 0) == True:
                if ((radius in [xc, yc, zc]) == False and (-radius in [xc, yc, zc]) == False) == True:
                    a.append(azimuthal)
                    b.append(polar)  # (correct_polar_in_python(polar))
                    x_new.append(xc)
                    y_new.append(yc)
                    z_new.append(zc)
        a = np.array(a)
        b = np.array(b)

        return a, b, x_new, y_new, z_new

    #####################################################################
    def polyhedra_octahedral_old(self, radius):

        check_distance = math.sqrt(radius + radius) + 0.1
        x = [radius, -radius, 0, 0, 0, 0]
        y = [0, 0, radius, -radius, 0, 0]
        z = [0, 0, 0, 0, radius, -radius]

        for times in range(3):
            x_new = []
            y_new = []
            z_new = []
            for i in range(len(x)):
                x_new.append(x[i])
                y_new.append(y[i])
                z_new.append(z[i])
                for j in range(i, len(x), 1):
                    distance = math.sqrt(((x[i] - x[j]) ** 2) + ((y[i] - y[j]) ** 2) + ((z[i] - z[j]) ** 2))
                    if (distance <= check_distance):
                        xc, yc, zc = self.vector_weight(x[i], y[i], z[i], x[j], y[j], z[j], 0.5)
                        x_new.append(xc)
                        y_new.append(yc)
                        z_new.append(zc)

            x = x_new.copy()
            y = y_new.copy()
            z = z_new.copy()

            xyz = []

            for i in range(len(x)):
                xyz.append([x[i], y[i], z[i]])
            xyz = np.array(xyz)

            xyz_new = []
            x_new = []
            y_new = []
            z_new = []
            for i in range(len(xyz)):
                for ch in range(len(xyz)):
                    if ((xyz[i] == xyz[ch]).all() and (self.is_vector_in_array(xyz[i], xyz_new) == False)) == True:
                        xyz_new.append(xyz[i])
                        dis2 = Phase2.distance_3d([0, 0, 0], xyz[i])
                        # print(xyz[i])
                        if (dis2 > radius / 2):
                            # ax.plot3D(xyz[i,0],xyz[i,1],xyz[i,2], 'r*', markersize=markersize_value)
                            x_new.append(xyz[i, 0])
                            y_new.append(xyz[i, 1])
                            z_new.append(xyz[i, 2])

            x = x_new
            y = y_new
            z = z_new
            x_new = []
            y_new = []
            z_new = []

            a = []
            b = []
            for i in range(len(x)):
                azimuthal, polar = self.Spherical_Coordinates_V2(x[i], y[i], z[i])
                xc, yc, zc = self.Cartesian_Coordinates(azimuthal, polar, radius)
                if (azimuthal < math.pi and azimuthal >= 0) == True:
                    if ((radius in [xc, yc, zc]) == False and (-radius in [xc, yc, zc]) == False) == True:
                        a.append(azimuthal)
                        b.append(polar)  # (correct_polar_in_python(polar))
                        x_new.append(xc)
                        y_new.append(yc)
                        z_new.append(zc)
        a = np.array(a)
        b = np.array(b)

        return a, b, x_new, y_new, z_new

    #####################################################################
    def check_number_of_exist(self, start, array, value):
        count = 0
        for ch in range(start, len(array), 1):
            if value == array[ch]:
                count += 1

        return count

    #####################################################################
    def check_if_visited(self, x, y, value_x, value_y, visited):
        for ch in range(0, len(x), 1):
            if value_x == x[ch] and value_y == y[ch]:
                return visited[ch]
        return 0

    #####################################################################
    def neighborhood_points(self, C_X, C_Y, change_c_x, change_c_y, radius, convex_hull, test_break, visited):
        list_x = []
        list_y = []
        condition = False
        if (len(convex_hull) >= 3):
            condition = True
        for i in range(len(C_X)):
            condition2 = True
            dist = math.sqrt(((change_c_x - C_X[i]) ** 2) + ((change_c_y - C_Y[i]) ** 2))
            if (change_c_x != C_X[i]):
                if (dist <= radius):
                    if (condition):
                        if ((C_X[int(convex_hull[test_break - 2])] == C_X[int(convex_hull[test_break])])
                            and (C_X[int(convex_hull[test_break - 1])] == C_X[i])) == True:
                            condition2 = False
                    if (condition2):
                        list_x.append(C_X[i])
                        list_y.append(C_Y[i])
                        visited[i] += 1
            elif (change_c_x == C_X[i]):
                visited[i] += 1

        c_x = np.array(list_x)
        c_y = np.array(list_y)

        return c_x, c_y, visited

    #####################################################################
    def check_if_flip_is_needed(self, C_X2, C_Y2, b_x, b_y, c_x, c_y, check_max_x_axes, check_min_x_axes, if_max_x_is_checked_first, visited, flip):
        count_quadrant_1 = 0
        count_quadrant_2 = 0
        count_quadrant_3 = 0
        count_quadrant_4 = 0

        is_unvisited_1 = 0
        is_unvisited_2 = 0
        is_unvisited_3 = 0
        is_unvisited_4 = 0
        for i in range(len(c_x)):
            line_B, quadrant, is_unvisited = self.get_angle_v4(b_x, b_y, c_x[i], c_y[i], C_X2, C_Y2, visited)
            if (quadrant == 1):
                count_quadrant_1 += 1
            if (quadrant == 2):
                count_quadrant_2 += 1
            if (quadrant == 3):
                count_quadrant_3 += 1
            if (quadrant == 4):
                count_quadrant_4 += 1

            if (is_unvisited == 1):
                is_unvisited_1 += 1
            if (is_unvisited == 2):
                is_unvisited_2 += 1
            if (is_unvisited == 3):
                is_unvisited_3 += 1
            if (is_unvisited == 4):
                is_unvisited_4 += 1

        if (check_max_x_axes > 0) and (count_quadrant_2 == 0) and (count_quadrant_3 == 0):
            return 1
        elif (check_max_x_axes > 0) and (is_unvisited_1 > 0):
            return 11
        elif (check_max_x_axes > 0) and (flip == 1 or flip == 11) and (count_quadrant_4 > 0 or count_quadrant_1 > 0):
            return 1
        elif (check_max_x_axes > 0) and (count_quadrant_3 > 0):
            return -1
        elif (check_max_x_axes > 0):
            return -1

        if (check_min_x_axes > 0) and (count_quadrant_1 == 0) and (count_quadrant_4 == 0):
            return -1
        elif (check_min_x_axes > 0) and (is_unvisited_3 > 0):
            return -11
        elif (check_min_x_axes > 0) and (flip == -1 or flip == -11) and (count_quadrant_3 > 0 or count_quadrant_2 > 0):
            return -1
        elif (check_min_x_axes > 0) and (count_quadrant_1 > 0):
            return 1
        elif (check_min_x_axes > 0):
            return 1

    #####################################################################
    def check_intersect(self, p1, p2, p3):
        val = (p1[1] - p2[1]) * (p3[0] - p2[0]) - (p1[0] - p2[0]) * (p3[1] - p2[1])
        # val = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
        if val > 0:
            return 1
        elif val == 0:
            return 0
        elif val < 0:
            return -1

    #####################################################################
    def Convex_Hull_JM(self, data, radius):
        convex_hull = []

        index_C_Y = np.argmin(data[:, 1])
        convex_hull.append(index_C_Y)
        index_c = index_C_Y
        b_x = data[index_c, 0]
        b_y = data[index_c, 1]
        a_x = data[index_c, 0] - 1
        a_y = data[index_c, 1]
        pre_a_x = a_x
        pre_a_y = a_y

        test_break = 0
        while (True):
            smallest_angle = 361
            line_A1 = self.get_angle3(b_x, b_y, a_x, a_y)
            c_i1 = self.check_intersect([a_x, a_y], [b_x, b_y], [pre_a_x, pre_a_y])
            dis = Phase2.distance_2d([a_x, a_y], [b_x, b_y])
            count_me = 0
            for c in range(len(data[:, 0])):
                dis = Phase2.distance_2d([data[c, 0], data[c, 1]], [b_x, b_y])
                if ((dis <= radius) and (dis != 0)):
                    count_me += 1
            for c in range(len(data[:, 0])):
                dis = Phase2.distance_2d([data[c, 0], data[c, 1]], [b_x, b_y])
                if ((dis <= radius) and (dis != 0) and (self.check_number_of_exist(2, convex_hull, c) <= 2)) == True:
                    line_B = self.get_angle3(b_x, b_y, data[c, 0], data[c, 1])
                    c_i2 = self.check_intersect([a_x, a_y], [data[c, 0], data[c, 1]], [pre_a_x, pre_a_y])
                    c_i3 = self.check_intersect([data[c, 0], data[c, 1]], [pre_a_x, pre_a_y], [b_x, b_y])
                    c_i4 = self.check_intersect([data[c, 0], data[c, 1]], [a_x, a_y], [b_x, b_y])
                    if (c_i1 + c_i2) != 0 or (c_i3 + c_i4) != 0 or test_break <= 1:
                        sub = line_B - line_A1
                        if (sub >= 0):
                            test_angle = sub
                        else:
                            test_angle = 360 + sub

                        if (smallest_angle >= test_angle and self.check_number_of_exist(2, convex_hull, c) <= 2 and (
                                count_me == 1 or test_angle != 0)):  # and (test_angle != 0):

                            smallest_angle = test_angle
                            convex_hull_last_input = c

            index_c = convex_hull_last_input
            pre_a_x = a_x
            pre_a_y = a_y
            a_x = b_x
            a_y = b_y
            b_x = data[index_c, 0]
            b_y = data[index_c, 1]

            if (test_break == 0):
                convex_hull_1 = index_C_Y
                convex_hull_2 = index_c
            test_break += 1

            if ((test_break == 1000) or
                    (((test_break > 1) and (a_x == data[convex_hull_1, 0]) and (
                            data[index_c, 0] == data[convex_hull_2, 0])) == True)
            ):
                break
            convex_hull.append(index_c)

        return convex_hull

    #####################################################################
    def Convex_Hull_P3(self, C_X2, C_Y2, radius):
        convex_hull = []
        data = np.zeros(len(C_X2)*2).reshape(len(C_X2), 2)
        data[:, 0] = C_X2
        data[:, 1] = C_Y2

        index_C_Y = np.argmin(data[:, 0])
        convex_hull.append(index_C_Y)

        index_c = index_C_Y
        b_x = data[index_c, 0]
        b_y = data[index_c, 1]
        a_x = data[index_c, 0] + 1
        a_y = data[index_c, 1]
        convex_hull_last_input = index_C_Y
        test_break = 0
        flip = 1

        if_max_x_is_checked_first = False
        if (index_C_Y == np.argmin(data[:, 0])):
            if_max_x_is_checked_first = True

        visited = np.zeros(len(C_X2))
        run_if_once = True
        while (True):
            #print(b_x)
            check_max_x_axes = self.check_number_of_exist(0, convex_hull, np.argmax(data[:, 0]))
            check_min_x_axes = self.check_number_of_exist(0, convex_hull, np.argmin(data[:, 0]))
            if (check_max_x_axes > 0) and (run_if_once):
                visited = np.zeros(len(C_X2))
                run_if_once = False
            c_x, c_y, visited = self.neighborhood_points(data[:, 0], data[:, 1], b_x, b_y, radius, convex_hull, test_break, visited)
            flip = self.check_if_flip_is_needed(C_X2, C_Y2, b_x, b_y, c_x, c_y, check_max_x_axes, check_min_x_axes, if_max_x_is_checked_first, visited, flip)

            smallest_angle = math.inf
            for c in range(len(data[:, 0])):
                if (self.check_number_of_exist(0, c_x, data[c, 0]) > 0):
                    if (len(c_x) < 2):
                        condition = True
                    elif (test_break < 3):
                        condition = True  # (check_number_of_exist(0, convex_hull, c) < 1)
                    else:
                        condition = True  # (check_number_of_exist(2, convex_hull, c) <= 5)

                    if ((condition)) == True:
                        line_B, quadrant, is_unvisited_ = self.get_angle_v4(b_x, b_y, data[c, 0], data[c, 1], C_X2, C_Y2, visited)
                        vector = self.Vectors_Calculation(data[c, 0], data[c, 1], b_x, b_y)
                        d__, cos_a = self.Angle_Between_Two_Vectors(vector, vector)
                        if (flip == 1) and ((quadrant == 1) or (quadrant == 4)):
                            line_B = vector[1] / d__
                        elif (flip == -1) and ((quadrant == 2) or (quadrant == 3)):
                            line_B = (-vector[1]) / d__
                        elif (flip == 11) and (is_unvisited_ == 1):
                            line_B = vector[1] / d__
                        elif (flip == -11) and (is_unvisited_ == 3):
                            line_B = (-vector[1]) / d__
                        else:
                            line_B = math.inf
                        if (smallest_angle >= line_B):  # and (test_angle != 0):
                            smallest_angle = line_B
                            convex_hull_last_input = c
            index_c = convex_hull_last_input
            a_x = b_x
            a_y = b_y
            b_x = data[index_c, 0]
            b_y = data[index_c, 1]

            if (test_break == 0):
                convex_hull_1 = index_C_Y
                convex_hull_2 = index_c
            test_break += 1

            if ((test_break == 1000) or
                    (((test_break > 1) and (a_x == data[convex_hull_1, 0]) and (
                            data[index_c, 0] == data[convex_hull_2, 0])) == True)
            ):
                break
            convex_hull.append(index_c)
        return convex_hull

    #####################################################################
    def Convex_Hull2(self, C_X, C_Y, graph, next_convex_hull):
        graph_previous = graph
        coordinate = 'y'
        while(True):
            convex_hull = []
            if (coordinate == 'y'):
                min_C_Y = max(C_Y)
                for c in range(len(C_X)):
                    if (C_Y[c] < min_C_Y) and (max(graph[c, :]) == 2):
                        min_C_Y = C_Y[c]
                        index_C_Y = c
            elif (coordinate == 'x'):
                min_C_Y = max(C_X)
                for c in range(len(C_X)):
                    if (C_X[c] < min_C_Y) and (max(graph[c, :]) == 2):
                        min_C_Y = C_X[c]
                        index_C_Y = c

            convex_hull.append(index_C_Y)

            #
            index_c = index_C_Y
            convex_hull_first_input = index_C_Y
            convex_hull_last_input = 0
            new_index_c = 0
            b_x = C_X[index_c]
            b_y = C_Y[index_c]
            a_x = C_X[index_c] - 1
            a_y = C_Y[index_c]

            test_break = 0
            main_break = 0
            while (convex_hull_first_input != convex_hull_last_input):
                larger_angle = 0
                for c in range(len(C_X)):
                    if ((graph[index_c, c] == 2) or (graph[index_c, c] == next_convex_hull)) and (
                            self.check_number_of_exist(1, convex_hull, c) <= 2):
                        graph[index_c, c] = next_convex_hull
                        graph[c, index_c] = next_convex_hull

                        line_A = self.get_angle3(b_x, b_y, a_x, a_y)
                        line_B = self.get_angle3(b_x, b_y, C_X[c], C_Y[c])
                        sub = line_A - line_B
                        if (sub >= 0):
                            test_angle =  sub
                        else:
                            test_angle = 360 + (sub)
                        if (larger_angle <= test_angle) and (340 > test_angle):
                            larger_angle = test_angle
                            convex_hull_last_input = c

                index_c = convex_hull_last_input
                a_x = b_x
                a_y = b_y
                b_x = C_X[index_c]
                b_y = C_Y[index_c]
                convex_hull.append(index_c)
                test_break += 1
                if (test_break == 1000):
                    break
            convex_hull_temporary = []
            for i in range(len(convex_hull)):
                if (i + 1 < len(convex_hull)):
                    if (convex_hull[i] != convex_hull[i + 1]):
                        convex_hull_temporary.append(convex_hull[i])
            convex_hull_temporary.append(convex_hull[0])
            convex_hull_test = convex_hull_temporary
            if (len(C_X) >= 3 and len(convex_hull_test) <= 3 and coordinate == 'y') == True:
                coordinate = 'x'
                graph = graph_previous
            else:
                break

        return convex_hull

    #####################################################################
    def Convex_Hull_Check(self, C_X, C_Y, graph):
        convex_hull = []

        min_C_Y = max(C_Y)
        for c in range(len(C_X)):
            if (C_Y[c] < min_C_Y):
                min_C_Y = C_Y[c]
                index_C_Y = c

        convex_hull.append(index_C_Y)

        #
        index_c = index_C_Y
        convex_hull_first_input = index_C_Y
        convex_hull_last_input = -1
        b_x = C_X[index_c]
        b_y = C_Y[index_c]
        a_x = C_X[index_c] - 1
        a_y = C_Y[index_c]

        while (convex_hull_first_input != convex_hull_last_input):
            larger_angle = 0
            for c in range(len(C_X)):
                if (self.check_if_exist(1, convex_hull, c) == False):
                    line_A = self.get_angle(b_x, b_y, a_x, a_y)
                    line_B = self.get_angle(b_x, b_y, C_X[c], C_Y[c])
                    if (line_A - line_B >= 0):
                        test_angle = line_A - line_B
                    else:
                        test_angle = 360 + (line_A - line_B)
                    if (larger_angle < test_angle) and (300 > test_angle):
                        larger_angle = test_angle
                        convex_hull_last_input = c

            index_c = convex_hull_last_input
            a_x = b_x
            a_y = b_y
            b_x = C_X[index_c]
            b_y = C_Y[index_c]
            convex_hull.append(index_c)

        return convex_hull

    #####################################################################
    def get_angle(self, x1, y1, x2, y2):
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if (angle >= 0):
            return angle
        else:
            return 360 + angle

    #####################################################################
    def get_angle2(self, x1, y1, x2, y2):
        if ((x1<=x2) and (y1<=y2)) == True : # 1st quadrant
            angle = math.atan2(abs(y2 - y1), abs(x2 - x1))
        elif ((x1>=x2) and (y1<=y2)) == True :# 2nd quadrant
            angle = (math.pi/2) + math.atan2(abs(x2 - x1), abs(y2 - y1))
        elif ((x1>=x2) and (y1>=y2)) == True :# 3rd quadrant
            angle = (math.pi) + math.atan2(abs(y2 - y1), abs(x2 - x1))
        elif ((x1<=x2) and (y1>=y2)) == True :# 4th quadrant
            angle = (3*math.pi/2) + math.atan2(abs(x2 - x1), abs(y2 - y1))
        else:
            print("what")

        return math.degrees(angle)

    #####################################################################
    def get_angle3(self, x1, y1, x2, y2):
        if ((x1<=x2) and (y1<=y2)) == True : # 1st quadrant
            q=0
        elif ((x1>=x2) and (y1<=y2)) == True :# 2nd quadrant
            q=1
        elif ((x1>=x2) and (y1>=y2)) == True :# 3rd quadrant
            q=2
        elif ((x1<=x2) and (y1>=y2)) == True :# 4th quadrant
            q=3

        angle = (q*math.pi/2) + math.atan2((abs(y2 - y1) / abs(x2 - x1))**((-1)**q),1)

        return math.degrees(angle)

    #####################################################################
    def check_if_exist(self, start, array, value):
        for ch in range(start, len(array), 1):
            if value == array[ch]:
                return True
        return False

    #####################################################################
    def Convex_Hull(self, C_X, C_Y, graph):
        convex_hull = []

        for c in range(len(C_X)):
            if (sum(graph[c, :]) == 1):
                check_other_point = np.argmax(graph[c, :])
                if (sum(graph[check_other_point, :]) == 1):
                    graph[c, check_other_point] = -1
                    graph[check_other_point, c] = -1

        min_C_Y = max(C_Y)
        for c in range(len(C_X)):
            if (C_Y[c] < min_C_Y) and (sum(graph[c, :]) > 0):
                min_C_Y = C_Y[c]
                index_C_Y = c

        convex_hull.append(index_C_Y)

        #
        index_c = index_C_Y
        convex_hull_first_input = index_C_Y
        convex_hull_last_input = 0
        b_x = C_X[index_c]
        b_y = C_Y[index_c]
        a_x = C_X[index_c] - 1
        a_y = C_Y[index_c]

        while (convex_hull_first_input != convex_hull_last_input):
            larger_angle = 0
            for c in range(len(C_X)):
                if (graph[index_c, c] == 1) and (self.check_if_exist(1, convex_hull, c) != True):
                    line_A = self.get_angle(b_x, b_y, a_x, a_y)
                    line_B = self.get_angle(b_x, b_y, C_X[c], C_Y[c])
                    if (line_A - line_B >= 0):
                        test_angle = line_A - line_B
                    else:
                        test_angle = 360 + (line_A - line_B)
                    if (larger_angle < test_angle) and (270 > test_angle):
                        larger_angle = test_angle
                        convex_hull_last_input = c

            index_c = convex_hull_last_input
            a_x = b_x
            a_y = b_y
            b_x = C_X[index_c]
            b_y = C_Y[index_c]
            convex_hull.append(index_c)

        return graph, convex_hull

    #####################################################################
    def create_folder(self, N_PHT, save_path=""):
        filename = save_path + "results/run/" + str(N_PHT) + "/1.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("FOOBAR")

    #####################################################################
    def angles(self, number_of_angles):
        a = np.zeros((number_of_angles))

        for i in range(number_of_angles):
            a[i] = math.pi / number_of_angles * i

        return a

    #####################################################################
    def check_input(self, message):
        while True:
            try:
                my_input = int(input(message))
            except ValueError:
                #print("Not an number!")
                continue
            else:
                return my_input

    #########################################
    def check_angle(self, angle):
        if (angle >= (math.pi * 2)):
            return angle - (math.pi * 2)
        elif (angle <= 0):
            return angle + (math.pi * 2)
        return angle

    #####################################################################
    def check_opposite_angle(self, angle):
        angle = angle + math.pi
        return self.check_angle(angle)

    #####################################################################
    def check_both_directions(self, angle_convex_hull, a):
        if (angle_convex_hull > (math.pi * 3 / 2)) and (a < math.pi / 2):
            angle_difference = a + ((math.pi * 2) - angle_convex_hull)
        elif (angle_convex_hull < (math.pi / 2)) and (a > math.pi * 3 / 2):
            angle_difference = ((math.pi * 2) - a) + angle_convex_hull
        else:
            angle_difference = abs(a - angle_convex_hull)
            if (angle_difference > math.pi):
                if (angle_convex_hull > a):
                    angle_difference = a + ((math.pi * 2) - angle_convex_hull)
                else:
                    angle_difference = ((math.pi * 2) - a) + angle_convex_hull
        return angle_difference

    #####################################################################
    def segmentation(self, len_C_X, graph):
        graph_number = 1
        len_Q = int(len_C_X / 3)
        return_Q = np.ones(len_C_X * len_C_X).reshape(len_C_X, len_C_X)
        return_Q *= -1

        for q in range(len_C_X):
            if (self.check_if_exist(0, graph[q, :], 1)):
                graph_number += 1
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
                    while (True):
                        z += 1
                        if (z >= len(Q)):
                            break
                        for j in range(len_C_X):
                            if (graph[Q[z], j] == 1):
                                graph[Q[z], j] = graph_number
                                graph[j, Q[z]] = graph_number
                                if (self.check_if_exist(0, Q, j) == False):
                                    Q.append(j)
                # if (graph_number == 2):
                for i in range(len(Q)):
                    return_Q[index_Q, i] = Q[i]

        return graph_number, graph, return_Q

    #####################################################################
    def this_PCA(self, C_X, C_Y, C_Z):
        C_XYZ = np.zeros(len(C_X) * 3).reshape(len(C_X), 3)

        for c in range(len(C_X)):
            C_XYZ[c] = [C_X[c], C_Y[c], C_Z[c]]

        pca = PCA()
        pca = pca.fit_transform(C_XYZ)
        return pca

    #####################################################################
    def find_a_b(self, C_X, a, Hough_Accumulate):
        temperary_Hough_Accumulate = np.zeros(len(C_X) * len(a) * len(a)).reshape(len(C_X), len(a), len(a))
        all_max_b = np.zeros(len(a))
        index_b = np.zeros(len(a))

        max_a = np.zeros(len(C_X))
        max_b = np.zeros(len(C_X))

        for c in range(len(C_X)):
            end_len_a = len(a)
            start_len_a = 0
            for i in range(len(a)):
                temperary_Hough_Accumulate[c, i, :] = Hough_Accumulate[c, start_len_a:end_len_a]
                all_max_b[i] = max(temperary_Hough_Accumulate[c, i, :])
                index_b[i] = np.argmax(temperary_Hough_Accumulate[c, i, :])
                start_len_a = end_len_a
                end_len_a = end_len_a + len(a)

            max_b[c] = np.argmax(all_max_b)
            max_a[c] = index_b[np.argmax(all_max_b)]

        return max_a, max_b, temperary_Hough_Accumulate

    #####################################################################
    def plot_line_3d_V2(self, x, y, z, a, b, length, ax):
        endx = x + (length * (math.cos(a) * math.sin(b)))
        endy = y + (length * (math.sin(a) * math.sin(b)))
        endz = z + (length * (math.cos(b)))
        return endx, endy, endz

    #####################################################################
    def check_repetitive(self, array, value):
        count_number_of_exist = 0
        for ch in range(1, len(array), 1):
            if value == array[ch]:
                count_number_of_exist = count_number_of_exist + 1
        return count_number_of_exist

    #####################################################################
    def calculate_distance(self, the_first_potential_end_index, the_second_potential_end_index, convex_hull, edge_distance):
        start_c = convex_hull.index(the_first_potential_end_index)
        end_c = convex_hull.index(the_second_potential_end_index)
        count_number_of_exist = self.check_repetitive(convex_hull, convex_hull[start_c])
        count_convex_hull = 0
        count = 0

        while (count_convex_hull < count_number_of_exist):
            count += edge_distance[start_c]

            start_c += 1
            if (start_c == len(convex_hull)):
                start_c = 0
            if (convex_hull[start_c] == convex_hull[end_c]):
                count_convex_hull += 1

        return count

    #####################################################################
    def prior_V4(self, number_of_r, number_of_angles, change_c_x, change_c_y, change_c_z, distance_of_nearest_neighbors, a, b, x_polyhedra, y_polyhedra, z_polyhedra):
        number_of_alpha_angles = number_of_angles

        r = np.linspace(-1, 1, (number_of_r * 2) + 1)

        r_cos_a_sin_b = np.zeros((number_of_alpha_angles, len(r)))
        r_sin_a_sin_b = np.zeros((number_of_alpha_angles, len(r)))
        r_cos_b = np.zeros((number_of_alpha_angles, len(r)))

        for i in range(len(x_polyhedra)):

            r_cos_a_sin_b[i] = r *  x_polyhedra[i] + change_c_x
            r_sin_a_sin_b[i] = r * y_polyhedra[i] + change_c_y
            r_cos_b[i] = r * z_polyhedra[i] + change_c_z

        return a, b, r, r_cos_a_sin_b, r_sin_a_sin_b, r_cos_b

    ######################################################################
