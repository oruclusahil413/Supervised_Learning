import numpy as np
import matplotlib.pyplot as plt
import random as random
import math

def compute_model_prediction(w, b, x_train):
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x_train[i] + b
    return f_wb



def the_gradient(w,b,x_train,y_train):
    m=x_train.shape[0]

    dj_dw=0
    dj_db=0

    for i in range(m):
        f_wb=w* x_train[i] + b
        dj_dw_i=(f_wb-y_train[i])*x_train[i]
        dj_db_i=(f_wb-y_train[i])

        dj_dw=dj_dw + dj_dw_i
        dj_db=dj_db + dj_db_i

    dj_dw=(1/m)*dj_dw
    dj_db=(1/m)*dj_db

    return dj_dw,dj_db



def gradient_descent(w_init,b_init,x_train,y_train
                     ,num_of_iterations,alpha,
                     the_gradient,cost_function):

    w=w_init
    b=b_init

    cost_list=[]
    wb_list=[]

    for i in range(num_of_iterations):

        dj_dw,dj_db=the_gradient(w,b,x_train,y_train)
        w=w- (alpha * dj_dw)
        b=b- (alpha * dj_db)

        if i<10000:
            cost_list.append(cost_function(w_init,b_init,x_train,y_train))
            wb_list.append(w,b)

        if i% math.ceil(num_of_iterations/10) == 0:
            print(f"Iteration {i}"
                  f"Cost: {cost_list[-1]}"
                  f"Weights: {w}"
                  f"Bias: {b}")

    return w,b,cost_list,wb_list



def cost_function(w, b, x_train, y_train):

    m = x_train.shape[0]
    cost_sum=0

    for i in range(m):

        f_wb=w* x_train[i] + b
        difference=f_wb - y_train[i]
        squared_difference=np.square(difference)
        cost_sum=cost_sum+squared_difference

    J_wb=cost_sum/(2*m)
    return J_wb




m = 100
w = 470
b = 270




"PLOTTING"

fig,axes = plt.subplots(1,2,figsize=(20, 6))

for idx in range(2):
    x_train = np.zeros(m)
    y_train = np.zeros(m)

    for j in range(m):
        x_train[j] = round(random.uniform(1, 20), 3)
        y_train[j] = round(random.uniform(1000, 10000), 2)

    x_train = np.sort(x_train)
    y_train = np.sort(y_train)

    prediction = compute_model_prediction(w, b, x_train)
    difference,lt=cost_function(w, b, x_train, y_train,prediction)

    # initialize parameters
    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2
    # run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(w_init, b_init, x_train, y_train,
                                                        iterations, tmp_alpha, cost_function, the_gradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

    axes[idx].scatter(x_train, y_train, color="red", label="y_train")
    axes[idx].plot(x_train, prediction, color="blue", label="prediction")

    for i in range(m):
      axes[idx].plot(
                     [x_train,x_train],
                     [y_train,prediction],
                     linestyle="dashed", color="green")
    axes[idx].set_title(f"Dataset #{idx + 1}")
    axes[idx].set_xlabel("x_train")
    axes[idx].set_ylabel("y")
    axes[idx].legend()

plt.tight_layout()
plt.show()

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()

