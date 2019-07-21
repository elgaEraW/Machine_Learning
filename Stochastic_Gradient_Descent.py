import numpy as np
import matplotlib.pyplot as plt

# Fix the number of features and the values of x and y

m = 100
x = 2* np.random.rand(m,1)
y = 4 + 3*x + np.random.randn(m,1)
plt.scatter(x,y)

# Set the hyper parameter of the learning schedule

n_epochs = 50
t0 , t1 = 5, 50

# Define the learning schedule function to find value of alpha after every iteration

def learning_scheduler(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)         # Select random theta values
X = np.c_[np.ones((m,1)),x]          # Set first value in X as 1

# Using the Gradient Descent Formula, find the value of theta for individual random values of x and y.

for epoch in range(n_epochs):
    for i in range(m):
        index = np.random.randint(m)
        xi = X[index:index+1]
        yi = y[index:index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_scheduler(epoch * m + i)
        theta = theta - eta*gradients
    X_new = np.c_[np.ones((2, 1)), [[0], [2]]]
    y_new = X_new.dot(theta)
    plt.plot(X_new, y_new)

print("theta = {}".format(theta))
plt.show()