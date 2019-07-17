import numpy as np
import matplotlib.pyplot as plt

# Fix the number of features and the values of x and y

m = 100
x = 2* np.random.rand(m,1)
y = 4 + 3*x + np.random.randn(m,1)
plt.scatter(x,y)            # Plot the points

# Set learning rate(alpha) and number of iterations

learning_rate = 0.1
iterations = 1000
theta = np.random.randn(2,1)            # Select any random theta value
X = np.c_[np.ones((m,1)),x]

# Using the Gradient Descent formula find the new values of theta and plot the new lines of predicted values

for i in range(iterations):
    gradient = 2/m * X.T.dot(X.dot(theta) - y)
    theta = theta - learning_rate * gradient
    X_new = np.array([[0], [2]])
    XN = np.c_[np.ones((2, 1)), X_new]
    Y = XN.dot(theta)
    plt.plot(XN, Y)

print('theta = {}'.format(theta))
plt.show()