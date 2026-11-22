import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as random

def computer_model_prediction(w,b,x_train):
    m=x_train.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x_train[i]+b

    return f_wb

m = 100
x_train = np.zeros(m)
y_train = np.zeros(m)
for i in range (0,100):
    a=random.uniform(1,20)
    x_train[i]=round(a,3)

    b=random.uniform(1000,10000)
    y_train[i]=round(b,2)

x_train=np.sort(x_train)
y_train=np.sort(y_train)

w=470
b=270

print(f"w{w}")
print(f"b{b}")
prediction=computer_model_prediction(w,b,x_train)
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,prediction, color='blue')
plt.show()
