# Import required modules

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Set no. of features and the polynomial

m = 100
X = 6 * np.random.rand(m,1) - 3
y = 2 * X**3 + 3*X**2 + 5*X + 6 + np.random.randn(m,1)
plt.scatter(X,y)

# Predict the polynomial using Polynomail Features and Linear Regression

poly_features = PolynomialFeatures(degree=3)
lin_reg = LinearRegression()
X_poly = poly_features.fit_transform(X)
lin_reg.fit(X_poly,y)

print('Values: {} {}'.format(lin_reg.intercept_,lin_reg.coef_[0]))
plt.show()
