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
from sklearn.linear_model import Ridge



from pandas import datetime
style.use('fivethirtyeight')

# import the dataset using pandas
# we remain the column
df = pd.read_csv("/home/lino/je/data_set1.csv",header=None, index_col=False, names = ['starttime','holidays','hour','load'])

#print the first 10 samples
print(df.head())
print(df.tail())

print("\n")


# Visualize the dataset
#plot the data
#df.plot()
#plt.legend()
#plt.show()


# check the target data
y = df['load']


# function to evaluate and plot a regressor on a given feature set
X = np.array(df.index.astype("int64").tolist()).reshape(-1, 1)


# Use the first 43800 data points as training, and the rest for testing
n_train = 52200




# function to evaluate and plot a regressor on a given feature set
def eval_on_features(features, target, regressor):
    #print("target:",target)
    #print("features:",features)
    #print('shape',df.starttime.shape)


    # split the given features into a training and a test set
    X_train, X_test = features[:n_train], features[n_train:61300]

    # split target array
    y_train, y_test = target[:n_train], target[n_train:61300]

    #X_train, X_test, y_train, y_test = train_test_split(data_dummies, df['load'], random_state=0)

    regressor.fit(X_train, y_train)
    print("Train-set R^2: {:.2f}".format(regressor.score(X_train, y_train)))
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    #print("feature importance:\n{}".format(regressor.feature_importances_))

    print("\n")
    
    
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))

    # visualize the prediction
    print("let's predict", y_pred)
    
    # visualize the train,test and prediction
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test,'-', label='test')
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")

    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Energy Load")
    plt.ylabel("points")
    plt.show()

# Evaluate the model using Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)



data_dummies = pd.get_dummies(df.drop('load', axis=1))
#print(data_dummies.head(5))


features = data_dummies

#print(features.head(2))
X_hour = features




#df['one'] = 1
df['starttime'] = pd.to_datetime(df.starttime)
data_starttime = df.set_index("starttime")
data_resampled = data_starttime.resample("1h").sum().fillna(0)

#print(".......",data_starttime)
#print("*******",data_resampled)
#print("vamos",data_resampled.index)
#print("data_resampled.hour", data_resampled.index.hour)


yy = data_starttime

X_hour = pd.get_dummies(data_starttime.drop('load', axis=1))

X_hour_x = np.array(data_resampled.index.hour.tolist()).reshape(-1, 1)



X_hour_week = np.hstack([np.array(data_resampled.index.dayofweek.tolist()).reshape(-1,1),
                         np.array(data_resampled.index.hour.tolist()).reshape(-1,1)])

#eval_on_features(X_hour_week, y, regressor)

from sklearn.linear_model import LinearRegression
eval_on_features(X_hour_week, y, LinearRegression())



from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()

eval_on_features(X_hour_week_onehot, y, Ridge())

from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

#X_hour_week_onehot = X_hour_week_onehot.toarray()


#X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)



lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)


