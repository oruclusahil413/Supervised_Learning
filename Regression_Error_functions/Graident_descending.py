import numpy as np
import math

def compute_model_prediction(w, b, x_train):
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x_train[i] + b
    return f_wb

def cost_function(w, b, x_train, y_train):
    m = x_train.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x_train[i] + b
        difference = f_wb - y_train[i]
        cost_sum += np.square(difference)
    total_cost = cost_sum / (2 * m)
    return total_cost

def the_gradient(w, b, x_train, y_train):
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x_train[i] + b
        dj_dw_i = (f_wb - y_train[i]) * x_train[i]
        dj_db_i = (f_wb - y_train[i])

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = (1 / m) * dj_dw
    dj_db = (1 / m) * dj_db

    return dj_dw, dj_db

def gradient_descent(w_init, b_init, x_train, y_train, num_of_iterations, alpha):
    w = w_init
    b = b_init

    cost_list = []
    wb_list = []

    for i in range(num_of_iterations):
        dj_dw, dj_db = the_gradient(w, b, x_train, y_train)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        if i < 10000:
            cost_list.append(cost_function(w, b, x_train, y_train))
            wb_list.append([w, b])

        if i % math.ceil(num_of_iterations / 10) == 0:
            print(f"Iteration {i} | Cost: {cost_list[-1]:.2f} | w: {w:.3f} | b: {b:.3f}")

    return w, b, cost_list, wb_list



