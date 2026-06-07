#####################################################################
# Copyright (c) 2023 Othman Alghamdi
# email: othhadi@gmail.com
# All rights reserved.

#####################################################################
import math
import numpy as np

class Polyhedra:

    #####################################################################
    def __init__(self, radius, level=1):
        self.radius = radius
        self.level = level+1
        #self.ax = ax
        #self.markersize_value = markersize_value

    #####################################################################
    def __add_vector_to_array_if_not_exist(self,vector,points_of_polyhedra,my_color):
        if ((vector in points_of_polyhedra.tolist()) == False):
            #self.ax.plot3D(vector[0], vector[1], vector[2], '.', color = my_color, markersize=self.markersize_value)
            return np.insert(points_of_polyhedra, len(points_of_polyhedra) * 3, vector).reshape(len(points_of_polyhedra) + 1, 3)
        return points_of_polyhedra

    #####################################################################
    def __vector_weight(self, p1, p2, dis):
        x = p1[0] + ((p2[0] - p1[0]) * dis)
        y = p1[1] + ((p2[1] - p1[1]) * dis)
        z = p1[2] + ((p2[2] - p1[2]) * dis)
        return x, y, z

    #####################################################################
    def __level_of_polyhedra(self, initial_points_of_polyhedra):
        points_of_polyhedra = []
        #0,2,4
        points_of_one_side = np.linspace(0, 1, self.level)
        side1 = []
        side2 = []
        for i in range(len(points_of_one_side)):
            side1.append(self.__vector_weight(initial_points_of_polyhedra[:, 0], initial_points_of_polyhedra[:, 2], points_of_one_side[i]))
            if(len(points_of_one_side)-1 > i):
                side2.append(self.__vector_weight(initial_points_of_polyhedra[:, 4], initial_points_of_polyhedra[:, 2], points_of_one_side[i]))

        side1 = np.array(side1)
        side2 = np.array(side2)
        for i in range(len(side1)):
            #self.ax.plot3D(side1[i,0], side1[i,1], side1[i,2], 'k.', markersize = self.markersize_value)
            points_of_polyhedra.append([side1[i,0], side1[i,1], side1[i,2]])
            if(len(points_of_one_side)-1 > i):
                #self.ax.plot3D(side2[i,0], side2[i,1], side2[i,2], 'g.', markersize = self.markersize_value)
                points_of_polyhedra.append([side2[i,0], side2[i,1], side2[i,2]])
            points_in_middle = np.linspace(0, 1, self.level-i)
            for j in range(1,len(points_in_middle)-1):
                if (len(points_of_one_side) - 1 > i):
                    x, y, z = (self.__vector_weight(side1[i, :], side2[i, :], points_in_middle[j]))
                    #self.ax.plot3D(x, y, z, 'r.', markersize = self.markersize_value)
                    points_of_polyhedra.append([x, y, z])

        points_of_polyhedra = np.array(points_of_polyhedra)
        for i in range(len(points_of_polyhedra)):
            points_of_polyhedra = self.__add_vector_to_array_if_not_exist([points_of_polyhedra[i,0], points_of_polyhedra[i,1]*-1, points_of_polyhedra[i,2]],points_of_polyhedra,'gray')
            points_of_polyhedra = self.__add_vector_to_array_if_not_exist([points_of_polyhedra[i,0]*-1, points_of_polyhedra[i,1]*-1, points_of_polyhedra[i,2]],points_of_polyhedra,'pink')
            points_of_polyhedra = self.__add_vector_to_array_if_not_exist([points_of_polyhedra[i,0]*-1, points_of_polyhedra[i,1], points_of_polyhedra[i,2]],points_of_polyhedra,'purple')
            points_of_polyhedra = self.__add_vector_to_array_if_not_exist([points_of_polyhedra[i,0], points_of_polyhedra[i,1]*-1, points_of_polyhedra[i,2]*-1],points_of_polyhedra,'orange')
            points_of_polyhedra = self.__add_vector_to_array_if_not_exist([points_of_polyhedra[i,0]*-1, points_of_polyhedra[i,1]*-1, points_of_polyhedra[i,2]*-1],points_of_polyhedra,'orange')
            points_of_polyhedra = self.__add_vector_to_array_if_not_exist([points_of_polyhedra[i,0]*-1, points_of_polyhedra[i,1], points_of_polyhedra[i,2]*-1],points_of_polyhedra,'orange')
            points_of_polyhedra = self.__add_vector_to_array_if_not_exist([points_of_polyhedra[i,0], points_of_polyhedra[i,1], points_of_polyhedra[i,2]*-1],points_of_polyhedra,'orange')
        return points_of_polyhedra

    #####################################################################
    def __polyhedra_octahedral_V4(self):
        # Calculating the hypotenuse of an isosceles triangle
        hypotenuse = self.radius * math.sqrt(2)

        check_distance = math.sqrt(self.radius + self.radius) + 0.1
        initial_points_of_polyhedra = np.zeros(18).reshape(3,6)
        initial_points_of_polyhedra[0] = [self.radius, -self.radius, 0, 0, 0, 0]
        initial_points_of_polyhedra[1] = [0, 0, self.radius, -self.radius, 0, 0]
        initial_points_of_polyhedra[2] = [0, 0, 0, 0, self.radius, -self.radius]

        points_of_polyhedra = self.__level_of_polyhedra(initial_points_of_polyhedra)

        return points_of_polyhedra

    #####################################################################
    def get_points_of_polyhedra(self):
        return self.__polyhedra_octahedral_V4()

#####################################################################
