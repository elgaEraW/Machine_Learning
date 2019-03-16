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

boston = datasets.load_boston()
print(boston.keys())
print(boston.DESCR)

boston_df = pd.DataFrame(data = boston.data, columns= boston.feature_names)
boston_df['PRICE'] = pd.Series(boston.target)

boston_df.info()
print(boston_df.head())
print(boston_df.describe())
#boston_df.plot(kind = 'scatter',x = 'CRIM',y = 'RAD')
#plt.show()

train , test = train_test_split(boston_df,test_size=0.2,random_state=42)
print(len(train),len(test))
data = boston_df.columns.drop('PRICE')
target = ['PRICE']
data_train = np.array(train[data])
data_test = np.array(test[data])
target_train = np.array(train[target])
target_test = np.array(test[target])

corr_matrix = boston_df.corr()
print(corr_matrix['RAD'].sort_values(ascending=False))

attributes = ['CRIM','RAD','TAX','ZN']
sns.pairplot(boston_df[attributes])

simputer = SimpleImputer(strategy='median')
simputer.fit(boston_df)
print(simputer.statistics_)

pipe = Pipeline([('simputer',SimpleImputer(strategy='median'))])

#boston_num = pipe.fit_transform(boston_df)
#print(boston_num.shape)

lin_reg = LinearRegression()
lin_reg.fit(data_train,target_train)

predicted_target = lin_reg.predict(data_test[:5])
print("Predicted: ",list(predicted_target))
print("Original Values: ",list(target_test[:5]))

lin_mse = mean_squared_error(predicted_target,target_test[:5])
lin_rmse = np.sqrt(lin_mse)
error = np.int(lin_rmse*1000)
print("ERROR: ",error,"$")
plt.show()