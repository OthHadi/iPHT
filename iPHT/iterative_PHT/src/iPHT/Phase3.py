import math
import random
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class Phase3:
    #####################################################################
    def parametric_equations(xq, yq, zq, a, b, c, t):
        x = xq + (a * t)
        y = yq + (b * t)
        z = zq + (c * t)

        return x, y, z

    #####################################################################
    def vector_weight(xq, yq, zq, xq2, yq2, zq2, w):
        w[0][0] = xq
        w[0][1] = xq2 - xq
        w[0][2] = 0.001

        w[1][0] = yq
        w[1][1] = yq2 - yq
        w[1][2] = 0.001

        w[2][0] = zq
        w[2][1] = zq2 - zq
        w[2][2] = 0.001
        return w

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
    def find_a_b(C_X, a, Hough_Accumulate):
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
    def Vectors_Calculation(x1, y1, z1, x2, y2, z2):
        return [(x1 - x2), (y1 - y2), (z1 - z2)]

    #####################################################################
    def Angle_Between_Two_Vectors(a, b):
        a_ = math.sqrt((a[0] ** 2) + (a[1] ** 2) + (a[2] ** 2))
        b_ = math.sqrt((b[0] ** 2) + (b[1] ** 2) + (b[2] ** 2))

        a_b = (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])
        if((a_ * b_) != 0):
            cos_a = a_b / (a_ * b_)
        else:
            cos_a = 0
        angle = math.acos(cos_a)

        if (angle > math.pi / 2):
            angle = math.pi - angle

        return cos_a, angle

    #####################################################################
    def C01234_V2(x0, x1, x2, y0, y1, y2, z0, z1, z2, x, y, z):
        C4 = (x2 ** 2) + (y2 ** 2) + (z2 ** 2)
        C3 = 2 * x1 * x2 + 2 * y1 * y2 + 2 * z1 * z2
        C2 = (x1 ** 2) + (y1 ** 2) + (z1 ** 2) + (2 * x2 * (x0 - x)) + (2 * y2 * (y0 - y)) + (2 * z2 * (z0 - z))
        C1 = (2 * x1 * (x0 - x)) + (2 * y1 * (y0 - y)) + (2 * z1 * (z0 - z))
        C0 = 0  # ((a1-xq)**2)+((a2-yq)**2)+((a3-zq)**2)

        return C0, C1 * 1, C2 * 2, C3 * 3, C4 * 4

    #####################################################################
    def C01234(a1, b1, c1, a2, b2, c2, a3, b3, c3, xq, yq, zq):
        weights = np.zeros(3 * 3).reshape(3, 3)
        print(a1)

        weights[0][0] = a1
        weights[0][1] = b1
        weights[0][2] = c1
        weights[1][0] = a2
        weights[1][1] = b2
        weights[1][2] = c2
        weights[2][0] = a3
        weights[2][1] = b3
        weights[2][2] = c3
        point = [xq, yq, zq]

        C4 = 4 * (np.sum(weights[:, 2] ** 2))
        C3 = 3 * np.sum(2 * (weights[:, 0] * weights[:, 1]))
        C2 = 2 * (np.sum(weights[:, 1] * weights[:, 1]) + np.sum((2 * weights[:, 2]) * (weights[:, 0] - point)))
        C1 = np.sum((2 * weights[:, 1]) * (weights[:, 0] - point))

        return 0, C1, C2, C3, C4

    #####################################################################
    def curve_equations(a1, b1, c1, a2, b2, c2, a3, b3, c3, t):
        x = (a1 * (t ** 0)) + (b1 * (t ** 1)) + (c1 * (t ** 2))
        y = (a2 * (t ** 0)) + (b2 * (t ** 1)) + (c2 * (t ** 2))
        z = (a3 * (t ** 0)) + (b3 * (t ** 1)) + (c3 * (t ** 2))

        return x, y, z

    def find_t_on_corve(x_best, t):
        x = x_best[0][0] + (x_best[0][1] * t) + (x_best[0][2] * (t ** 2))
        y = x_best[1][0] + (x_best[1][1] * t) + (x_best[1][2] * (t ** 2))
        z = x_best[2][0] + (x_best[2][1] * t) + (x_best[2][2] * (t ** 2))

        return x, y, z
    #####################################################################
    def derivative_curve_equations(b1, c1, b2, c2, b3, c3, t):
        v1 = np.zeros(3)

        v1[0] = b1 + (2 * c1 * t)
        v1[1] = b2 + (2 * c2 * t)
        v1[2] = b3 + (2 * c3 * t)

        return v1

    #####################################################################
    def func(weights, t, root):
        tp = np.power(root, t)

        out = np.dot(weights, tp).T

        # Whether inputroot is a single number or an array, return output in correct form
        if (out.shape[0] == 1):
            return out[0]
        else:
            return out

    #####################################################################
    def get_t(w, xq, yq, zq):
        point = [xq, yq, zq]
        C4 = 4 * (np.sum(w[:, 2] ** 2))
        C3 = 3 * np.sum(2 * (w[:, 0] * w[:, 1]))
        C2 = 2 * (np.sum(w[:, 1] * w[:, 1]) + np.sum((2 * w[:, 2]) * (w[:, 0] - point)))
        C1 = np.sum((2 * w[:, 1]) * (w[:, 0] - point))

        C0, C1, C2, C3, C4 = Phase3.C01234_V2(w[0][0], w[0][1], w[0][2], w[1][0], w[1][1], w[1][2], w[2][0], w[2][1], w[2][2],
                                       xq,
                                       yq, zq)

        d_sqrd_derivative = Polynomial([C1, C2, C3, C4])
        roots = d_sqrd_derivative.roots()
        three_real = True
        real_index = []

        for j in range(0, len(roots)):
            if np.iscomplex(roots[j]):
                three_real = False
            else:
                real_index.append(j)

        # If all roots are real, pick the one that generates the closest point to our current point
        if three_real:
            root = roots
            closest_points = Phase3.func(w, np.array([[0], [1], [2]]), root)
            t_dists = []
            for k in range(0, len(root)):
                t_dists.append(math.dist(point, closest_points[k]))

            t_dists = np.array(t_dists)
            best_root = root[t_dists.argmin()]

        # If there is only one, use that to calculate the distance
        else:
            best_root = roots[real_index]

        return best_root.astype(float)

    #####################################################################
    def vectors_of_q(xq, yq, zq, ang):
        vector = np.zeros(3 * len(ang) * len(ang)).reshape(len(ang) * len(ang), 3)
        index = 0

        for i in range(len(ang)):
            for j in range(len(ang)):
                xq2, yq2, zq2 = Phase3.plot_line_3d_V2(xq, yq, zq, ang[i], ang[j], 0.1)
                vector[index, :] = Phase3.Vectors_Calculation(xq, yq, zq, xq2, yq2, zq2)
                index += 1

        return vector

    #####################################################################
    def Curve_Loss_Function2(w, a1, b1, c1, a2, b2, c2, a3, b3, c3, xq, yq, zq, ang, vectors, Hough_Accumulate, xxx, yyy, zzz, last_lp, Use_l, k):
        probabilities = np.zeros(len(xq))
        distances = np.zeros(len(xq))

        for i in range(len(xq)):
            point = [xq[i], yq[i], zq[i]]
            C4 = 4 * (np.sum(w[:, 2] ** 2))
            C3 = 3 * np.sum(2 * (w[:, 0] * w[:, 1]))
            C2 = 2 * (np.sum(w[:, 1] * w[:, 1]) + np.sum((2 * w[:, 2]) * (w[:, 0] - point)))
            C1 = np.sum((2 * w[:, 1]) * (w[:, 0] - point))

            C0, C1, C2, C3, C4 = Phase3.C01234_V2(a1, b1, c1, a2, b2, c2, a3, b3, c3, xq[i], yq[i], zq[i])

            # Put the coefficients into the numPy polynomial class and get the roots
            d_sqrd_derivative = Polynomial([C1, C2, C3, C4])
            roots = d_sqrd_derivative.roots()
            three_real = True
            real_index = []

            # Check if there are three real roots or one
            for j in range(0, len(roots)):
                if np.iscomplex(roots[j]):
                    # print("not real roots = " + str(roots[j]))
                    three_real = False
                else:
                    # print("roots = " + str(roots[j]))
                    real_index.append(j)

            distance = 0
            best_root = 0

            # If all roots are real, pick the one that generates the closest point to our current point
            if three_real:
                root = roots
                closest_point = Phase3.func(w, np.array([[0], [1], [2]]), root)
                t_dists = []
                for k in range(0, len(root)):
                    t_dists.append(math.dist(point, closest_point[k]))

                t_dists = np.array(t_dists)
                distance = t_dists[t_dists.argmin()]
                best_root = root[t_dists.argmin()]

            # If there is only one, use that to calculate the distance
            else:
                root = roots[real_index]
                closest_point = Phase3.func(w, np.array([[0], [1], [2]]), root)
                distance = math.dist(point, closest_point)
                best_root = roots[real_index]

            distances[i] = distance ** 2

            # calculate the angle between the two vectors
            # xq2,yq2,zq2 = plot_line_3d_V2(xq[i], yq[i], zq[i], ang[int(max_a[i])], ang[int(max_b[i])], 0.1)
            # v2 = Vectors_Calculation(xq[i], yq[i], zq[i], xq2, yq2, zq2)

            v1_t = Phase3.derivative_curve_equations(b1, c1, b2, c2, b3, c3, best_root.astype(float))
            #v1_t = Phase3.curve_equations(a1, b1, c1, a2, b2, c2, a3, b3, c3, best_root.astype(float))
            xv, yv, zv = Phase3.parametric_equations(xq[i], yq[i], zq[i], v1_t[0], v1_t[1], v1_t[2], 0.5)

            v1 = Phase3.Vectors_Calculation(xv, yv, zv, xq[i], yq[i], zq[i])
            # print("v1"+str(v1))
            angle_Between_Vectors = np.zeros(len(vectors))
            for e in range(len(vectors)):
                cos_a, angle_Between_Vectors[e] = Phase3.Angle_Between_Two_Vectors(v1_t, vectors[e])

            probabilities[i] = Hough_Accumulate[i, np.argmin(angle_Between_Vectors)]

        ld = 1
        sum_probabilities = np.sum(probabilities) / len(xq)
        sum_distances = np.sum(distances) / len(xq)

        if(Use_l):
            la = (sum_distances / sum_probabilities) * 0.1
        else:
            la = last_lp
        loss_a = sum_probabilities * la
        loss_d = sum_distances * ld
        loss_function = loss_d - loss_a
        return sum_probabilities, sum_distances, loss_function, la

    #####################################################################
    def weights(weight, k, k_max):
        all_weights = []

        rand = random.uniform(0, 1)
        amount = (1 - (rand ** (1 - (k / k_max))))

        for i in range(0, len(weight)):
            for j in range(0, len(weight[i])):
                new_weight = []
                new_weight = weight.copy()
                #new_weight[i][j] += amount
                new_weight[i][j] *= (amount + 1)
                all_weights.append(new_weight.copy())

                new_weight = weight.copy()
                #new_weight[i][j] -= amount
                new_weight[i][j] *= ((-amount)  + 1)
                all_weights.append(new_weight.copy())

        return all_weights[random.randint(0, len(all_weights) - 1)]

    #####################################################################
    def probability(loss_function, e_new, temperature):
        if (e_new < loss_function):
            p = 1
        else:
            p = math.exp((loss_function - e_new) / temperature)
        return p

    #####################################################################
    def kk(T, len_T_X_all, len_a, KK):
        K = np.zeros(len_T_X_all * len_a).reshape(len_T_X_all, len_a)
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

        return K

    #####################################################################
    def polynomial_regression3d(x, y, z, degree):
        # sort data to avoid plotting problems
        x, y, z = zip(*sorted(zip(x, y, z)))

        t = np.linspace(0, 1, len(x))
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        data_yz = np.array([x, y, z])
        data_yz = data_yz.transpose()

        polynomial_features = PolynomialFeatures(degree=degree)
        x_poly = polynomial_features.fit_transform(t[:, np.newaxis])

        model = LinearRegression()
        model.fit(x_poly, data_yz)
        y_poly_pred = model.predict(x_poly)

        coef = model.coef_
        intercept = model.intercept_

        coef[0, 0] = intercept[0]
        coef[1, 0] = intercept[1]
        coef[2, 0] = intercept[2]

        rmse = np.sqrt(mean_squared_error(data_yz, y_poly_pred))
        r2 = r2_score(data_yz, y_poly_pred)

        # plot
        return y_poly_pred[:, 0], y_poly_pred[:, 1], y_poly_pred[:, 2], coef
        # fig.set_dpi(150)

    ######################################################################
    def vectors_of_q(x_polyhedra, y_polyhedra, z_polyhedra):
        vector = np.zeros(3 * len(x_polyhedra)).reshape(len(x_polyhedra), 3)

        for i in range(len(x_polyhedra)):
            vector[i, :] = Phase3.Vectors_Calculation(x_polyhedra[i], y_polyhedra[i], z_polyhedra[i],0,0,0)

        return vector
    ######################################################################

    def x(t):
        return (t**2)+t+1

    def y(t):
        return np.sin(t)

    def z(t):
        return t

    def curve_length(x_func, y_func, z_func, t_start, t_end, num_points=1000):
        # Create an array of equally spaced t values between t_start and t_end
        t_values = np.linspace(t_start, t_end, num_points)

        # Calculate the differentials for each t value
        dt = t_values[1] - t_values[0]

        # Calculate the derivatives of x, y, and z with respect to t
        dx_dt = np.gradient(x_func(t_values), dt)
        dy_dt = np.gradient(y_func(t_values), dt)
        dz_dt = np.gradient(z_func(t_values), dt)

        # Calculate the squared derivatives
        squared_derivatives = dx_dt ** 2 + dy_dt ** 2 + dz_dt ** 2

        # Take the square root of the sum of squared derivatives
        square_root = np.sqrt(squared_derivatives)

        # Calculate the integral using the trapezoidal rule
        integral = np.trapz(square_root, t_values)

        return integral

    ###############################################################
    def curve_limit_distance(x_best, start, stop, num_points = 100):
        distance = 0
        x = np.zeros(num_points+1)
        y = np.zeros(num_points+1)
        z = np.zeros(num_points+1)
        par = np.linspace(start, stop, num_points)
        for i in range(len(par)):
            if (i+1 < len(par)):
                x[i] = x_best[0][0] + (x_best[0][1] * par[i]) + (x_best[0][2] * (par[i] ** 2))
                y[i] = x_best[1][0] + (x_best[1][1] * par[i]) + (x_best[1][2] * (par[i] ** 2))
                z[i] = x_best[2][0] + (x_best[2][1] * par[i]) + (x_best[2][2] * (par[i] ** 2))

                x2 = x_best[0][0] + (x_best[0][1] * par[i+1]) + (x_best[0][2] * (par[i+1] ** 2))
                y2 = x_best[1][0] + (x_best[1][1] * par[i+1]) + (x_best[1][2] * (par[i+1] ** 2))
                z2 = x_best[2][0] + (x_best[2][1] * par[i+1]) + (x_best[2][2] * (par[i+1] ** 2))
                x[num_points]=x2
                y[num_points]=y2
                z[num_points]=z2
                distance = distance + math.dist([x[i],y[i],z[i]],[x2,y2,z2])


        return distance, x,y,z
    ###############################################################
    def curve_limit_distance_new(x_best, start, stop, num_points = 100):
        distance = 0
        for i in np.linspace(start, stop, num_points):
            if (i+1 < num_points):
                x = x_best[0][0] + (x_best[0][1] * i) + (x_best[0][2] * (i ** 2))
                y = x_best[1][0] + (x_best[1][1] * i) + (x_best[1][2] * (i ** 2))
                z = x_best[2][0] + (x_best[2][1] * i) + (x_best[2][2] * (i ** 2))

                x2 = x_best[0][0] + (x_best[0][1] * i+1) + (x_best[0][2] * (i+1 ** 2))
                y2 = x_best[1][0] + (x_best[1][1] * i+1) + (x_best[1][2] * (i+1 ** 2))
                z2 = x_best[2][0] + (x_best[2][1] * i+1) + (x_best[2][2] * (i+1 ** 2))

                distance = distance + math.dist([x,y,z],[x2,y2,z2])

        return distance
