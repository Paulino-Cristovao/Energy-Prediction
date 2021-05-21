import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import os
import matplotlib
import numpy as np
import mglearn
from sklearn.model_selection import train_test_split
from datetime import date, timedelta
import holidays

from sklearn.model_selection import train_test_split
import data1

from pandas import datetime
style.use('fivethirtyeight')

# import the dataset using pandas
# i decide to remain the column
df = pd.read_csv("/home/lino/je/data_set12.csv",header=None, index_col=False, names = ['starttime','holidays','weather','hour','load'])


#print the first 10 samples
#print(df.head())

#plot the data
#df.plot()
#plt.legend()
plt.show()


# check the target data
y = df['load']


# function to evaluate and plot a regressor on a given feature set
X = np.array(df.index.astype("int64").tolist()).reshape(-1, 1)


# Use the first 43800 data points as training, and the rest for testing
n_train = 52584




# function to evaluate and plot a regressor on a given feature set
def eval_on_features(features, target, regressor):
    #print("target:",target)
    #print("features:",features)
    #print('shape',df.starttime.shape)


    # split the given features into a training and a test set
    X_train, X_test = features[:n_train], features[n_train:61343]

    # split target array
    y_train, y_test = target[:n_train], target[n_train:61343]


    regressor.fit(X_train, y_train)
    print("Train-set R^2: {:.2f}".format(regressor.score(X_train, y_train)))
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    #print("feature importance:\n{}".format(regressor.feature_importances_))


    print("\n")
    from sklearn.metrics import mean_squared_error
    #print("RMSE train: {:.2f}".format(mean_squared_error(X_train, y_train)))
   # print("RMSE test: {:.2f}".format(mean_squared_error(X_test, y_test)))

    
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    
    plt.figure(figsize=(10, 3))

    print("let's predict", y_pred)
    
    np.savetxt('forecast1.csv', np.rint(y_pred), delimiter=",")


    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test,'-', label='test')
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")

    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Load")
    plt.ylabel("points")
    plt.show()

    

# Evaluate the model using Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, min_samples_split = 2, verbose = 0, min_samples_leaf=100, n_jobs = -1, random_state=0)


data_dummies = pd.get_dummies(df.drop('load', axis=1))


features = data_dummies
X_hour = features


df['week'] = pd.to_datetime(df.starttime).dt.dayofweek
df['hour'] = pd.to_datetime(df.hour).dt.hour
df['month'] = pd.to_datetime(df.starttime).dt.month
df['daysofyear'] = pd.to_datetime(df.starttime).dt.dayofyear



df = pd.get_dummies(df.drop(["starttime","load","weather"], axis=1))
df["temp"] = data1.l

df = pd.get_dummies(df.drop('daysofyear',axis=1))


#eval_on_features(df, y, regressor)
df = df.values.reshape(-1,1)
print(df)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
#eval_on_features(df, y, LinearRegression())


enc = OneHotEncoder()
X_hour_week_hot = enc.fit_transform(df).toarray()
eval_on_features(df, y, Ridge())


"""

poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_hot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)
"""





