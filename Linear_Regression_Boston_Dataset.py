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


def load_dataset():
    boston = datasets.load_boston()
    print(boston.keys())
    print(boston.DESCR)
    return boston

def dataset_to_dataframe():
    boston_df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
    boston_df['PRICE'] = pd.Series(boston.target)
    return boston_df

def print_description():
    boston_df.info()
    print(boston_df.head())
    print(boston_df.describe())

#boston_df.plot(kind = 'scatter',x = 'CRIM',y = 'RAD')
#plt.show()

def split_train_test():
    train, test = train_test_split(boston_df, test_size=0.2, random_state=42)
    print(len(train), len(test))
    data = boston_df.columns.drop('PRICE')
    target = ['PRICE']
    data_train = np.array(train[data])
    data_test = np.array(test[data])
    target_train = np.array(train[target])
    target_test = np.array(test[target])
    return data_train,data_test,target_train,target_test

def corelation_matrix():
    corr = boston_df.corr()
    sns.heatmap(corr,cbar = True,annot=True,square=True)

def scatter_plot():
    attributes = ['CRIM','RAD','TAX','RM','PRICE']
    sns.pairplot(boston_df[attributes])

def imputer_median():
    simputer = SimpleImputer(strategy='median')
    simputer.fit(boston_df)
    #print(simputer.statistics_)

def pipeline():
    pipe = Pipeline([('simputer',SimpleImputer(strategy='median'))])

#boston_num = pipe.fit_transform(boston_df)
#print(boston_num.shape)

def linear_regression():
    lin_reg = LinearRegression()
    lin_reg.fit(data_train,target_train)

    predicted_target = lin_reg.predict(data_test)
    #print("Predicted: ",list(predicted_target))
    #print("Original Values: ",list(target_test))
    print(lin_reg.coef_)
    return predicted_target

def error():
    lin_mse = mean_squared_error(predicted_target,target_test)
    lin_rmse = np.sqrt(lin_mse)
    error = np.int(lin_rmse*1000)
    print("ERROR: ",error,"$")

boston = load_dataset()
boston_df = dataset_to_dataframe()
print_description()
data_train,data_test,target_train,target_test = split_train_test()
corelation_matrix()
scatter_plot()
imputer_median()
pipeline()
predicted_target = linear_regression()
error()
plt.show()
