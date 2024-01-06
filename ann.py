# Step 1: Importing Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
import math
from sklearn.metrics import mean_squared_error

# Step 2: Importing the dataset
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

# Step 5: Splitting the dataset into test train sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1000)
from keras.models import Sequential
from keras import layers

# Step 6: Training the model
input_dim = 1 # Number of features
model = Sequential()
model.add(layers.Dense(50, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=100, verbose=False,validation_data=(x_test, y_test), batch_size=16)

# Step 7:Making prediction
testPredict = model.predict(x_test)
y_test = y_test.reshape(-1,1)
testPredict = testPredict.reshape(-1,1)

# Step 8: Inverse scaling
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler1.fit(testPredict)
testPredict = scaler1.inverse_transform(testPredict)
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2.fit(y_test)
y_test = scaler2.inverse_transform(y_test)

# Step 9: Calcuating RMSE
testScore = math.sqrt(mean_squared_error(y_test, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# Step 10: Plotting the results
plt.scatter(x_test,y_test, color = 'red')
plt.scatter(x_test, testPredict, color = 'blue')
plt.title('NN - 2 Hidden Layer Prediction')
plt.xlabel('Num of days')
plt.ylabel('Forex rate')
plt.show()