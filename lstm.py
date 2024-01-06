# Step 1: Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Step 2: Importing the datasets
dataset = pd.read_csv("currency_exchange_rates.csv")
dataset = dataset[['Num_days','Indian Rupee']]

# Step 3: Replacing missing values
dataset = dataset.interpolate(method ='linear', limit_direction ='forward')
dataset.iloc[0,1] = dataset.iloc[1,1]

# Step 4: Feature Scaling
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
X = []
Y = []
for i in range(len(dataset)):
X.append(dataset[i][0])
Y.append(dataset[i][1])
X = np.array(X)
Y = np.array(Y)

# Step 5: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 1000)
Y_Train = Y_Train.reshape(-1,1)
Y_Test = Y_Test.reshape(-1,1)
X_Train = X_Train.reshape(-1,1)
X_Test = X_Test.reshape(-1,1)

# Step 6: Reshape inputs
X_Train = np.reshape(X_Train, (X_Train.shape[0], 1, X_Train.shape[1]))
X_Test = np.reshape(X_Test, (X_Test.shape[0], 1, X_Test.shape[1]))

# Step 7: Training the model
from keras.models import Sequential
from keras import layers

# Step 8: create and fit the LSTM model
model = Sequential()
model.add(LSTM(7, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_Train, Y_Train, epochs=100, batch_size=15, verbose=2)

# Step 9: Make prediction
testPredict = model.predict(X_Test)

# Step 10: Inverse scaling
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler1.fit(testPredict)
testPredict = scaler1.inverse_transform(testPredict)
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2.fit(Y_Test)
Y_Test = scaler2.inverse_transform(Y_Test)

# Step 11: Calculating RMSE
testScore = math.sqrt(mean_squared_error(Y_Test[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Step 12: Plotting results
plt.scatter(X_Test,Y_Test, color = 'red')
plt.scatter(X_Test, testPredict, color = 'blue')
plt.title('LSTM Results')
plt.xlabel('Num of days')
plt.ylabel('Forex rate')
plt.show()