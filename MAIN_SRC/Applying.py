import numpy as np
import matplotlib.pyplot as plt
import random
from Regression_model import compute_model_prediction, gradient_descent
import math

m = 100
w_init = 0
b_init = 0
iterations = 1000
alpha = 0.01

Graph_NUMBER=5
if Graph_NUMBER > 3:
    rows=math.ceil(Graph_NUMBER//3)+1
    fig, axes = plt.subplots(rows, 3, figsize=(20, 6))
else:
    fig, axes = plt.subplots(1, Graph_NUMBER, figsize=(20, 6))

axes=axes.flatten()

for idx in range(Graph_NUMBER):
    x_train = np.zeros(m)
    y_train = np.zeros(m)

    for j in range(m):
        x_train[j] = round(random.uniform(1, 20), 3)
        y_train[j] = round(random.uniform(1000, 10000), 2)

    x_train = np.sort(x_train)
    y_train = np.sort(y_train)

    w_final, b_final, cost_hist, wb_hist = gradient_descent(
        w_init, b_init, x_train, y_train, iterations, alpha
    )

    prediction = compute_model_prediction(w_final, b_final, x_train)

    axes[idx].scatter(x_train, y_train, color="red", label="y_train")
    axes[idx].plot(x_train, prediction, color="blue", label="prediction")

    for i in range(m):
        axes[idx].plot(
            [x_train[i], x_train[i]],
            [y_train[i], prediction[i]],
            linestyle="dashed", color="green"
        )

    axes[idx].set_title(f"Dataset #{idx + 1}")
    axes[idx].set_xlabel("x_train")
    axes[idx].set_ylabel("y")
    axes[idx].legend()

plt.tight_layout()
plt.show()

