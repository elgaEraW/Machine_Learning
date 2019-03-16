from sklearn import datasets
import pandas as pd
import numpy as np
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',100)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

boston = datasets.load_boston()
#print(boston.keys())
print(boston.DESCR)

boston_df = pd.DataFrame(data = boston.data, columns= boston.feature_names)
boston_df['PRICE'] = pd.Series(boston.target)

#boston_df.info()
#print(boston_df.head())
#print(boston_df.describe())
#boston_df.plot(kind = 'scatter',x = 'CRIM',y = 'RAD')
#plt.show()

train , test = train_test_split(boston_df,test_size=0.2,random_state=42)
#print(len(train),len(test))
data = boston_df.columns.drop('PRICE')
target = ['PRICE']
data_train = np.array(train[data])
data_test = np.array(test[data])
target_train = np.array(train[target])
target_test = np.array(test[target])

corr_matrix = boston_df.corr()
#print(corr_matrix['RAD'].sort_values(ascending=False))

attributes = ['CRIM','RAD','TAX','ZN']
sns.pairplot(boston_df[attributes])

simputer = SimpleImputer(strategy='median')
simputer.fit(boston_df)
#print(simputer.statistics_)

pipe = Pipeline([('simputer',SimpleImputer(strategy='median'))])

#boston_num = pipe.fit_transform(boston_df)
#print(boston_num.shape)

lin_reg = LinearRegression()
lin_reg.fit(data_train,target_train)

predicted_target_lin_reg = lin_reg.predict(data_test)
#print("Predicted: ",list(predicted_target_lin_reg))
#print("Original Values: ",list(target_test))

lin_mse = mean_squared_error(predicted_target_lin_reg,target_test)
lin_rmse = np.sqrt(lin_mse)
error_lin_reg = np.int(lin_rmse*1000)
print("ERROR LINEAR REGRESSION: ",error_lin_reg,"$")
#plt.show()

tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_train,target_train)

predicted_target_tree_reg = tree_reg.predict(data_test)
#print("Predicted: ",list(predicted_target_tree_reg))
#print("Original Values: ",list(target_test))

tree_mse = mean_squared_error(predicted_target_tree_reg,target_test)
tree_rmse = np.sqrt(tree_mse)
error_tree_reg = np.int(tree_rmse*1000)
print("ERROR TREE REGRESSION: ",error_tree_reg,"$")

forest_reg = RandomForestRegressor()
forest_reg.fit(data_train,target_train)

predicted_target_forest_reg = forest_reg.predict(data_test)
#print("Predicted: ",list(predicted_target_forest_reg))
#print("Original Values: ",list(target_test))

forest_mse = mean_squared_error(predicted_target_forest_reg,target_test)
forest_rmse = np.sqrt(forest_mse)
error_forest_reg = np.int(forest_rmse*1000)
print("ERROR FOREST_REGRESSION: ",error_forest_reg,"$")
