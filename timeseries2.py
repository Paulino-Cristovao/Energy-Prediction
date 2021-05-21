# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
#from sklearn.preprocessing import transform
# fix random seed for reproducibility
numpy.random.seed(7)

# Load dataset
df = pd.read_csv("/home/lino/je/data_set1.csv", header=None, index_col=False,\
                names = ['date','holiday','hour','load'])

dataset = df.load
dataset = dataset.astype('float32')

#print(dataset)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
dataset = dataset.values.reshape(-1, 1)

dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print(train.shape)
#print(test.shape)

look_back = 100

#look_back is the number of previous time step to use as input
#to predict the next time period
def create_dataset(dataset, look_back):
    dataX, dataY = [],[]
    for i in range(len(dataset) - look_back-100):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        #dataY.append(dataset[i + look_back: i + look_back + 24, 0])
        dataY.append(dataset[i + look_back, 0])

        #dataY= dataY[:24]

        #print(dataY)
        #print(len(dataY))
        #print(len(dataX))


    return numpy.array(dataX), numpy.array(dataY)


        

# reshape into X=t and Y=t+1


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)



# Reshape input to be [samples, tim steps and features]
trainX = numpy.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
testX = numpy.reshape(testX, (testX.shape[0],testX.shape[1],1))

# create and fit the LSTM network
from keras.layers import SimpleRNN
model = Sequential()
#model.add(LSTM(4, input_shape=(look_back, 24), activation = 'softmax'))
model.add(SimpleRNN(512, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=2, batch_size=100, verbose=2)



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#Invert predictions
# we invert the predictions before calculating errors scores to
# ensure that performance is reported in  the same unit as original data
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

#print("trainy",trainY.shape)
#print("trainPredict",trainPredict.shape)
#print("testY",testY.shape)
#print("testPredict",testPredict.shape)


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#print(trainPredict)
print("\n")
#print(testPredict)

print('********')
from sklearn.metrics import r2_score
print('Train Score: %.2f R^2' % r2_score(trainY, trainPredict, multioutput='variance_weighted'))
print('Test Score: %.2f R^2' % r2_score(testY, testPredict, multioutput='variance_weighted '))
"""
# shift train predictions for plotting
# 
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

#shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :]= numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :]=testPredict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
"""
