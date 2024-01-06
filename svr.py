# Step 1: Importing Dependencies
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Step 2: Importing the datasets
dataset = pd.read_csv("currency_exchange_rates.csv")
dataset = dataset[['Num_days','Indian Rupee']]

# Step 3: Replacing missing values
dataset = dataset.interpolate(method ='linear', limit_direction ='forward')
dataset.iloc[0,1] = dataset.iloc[1,1]

#Feature Scaling
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

# Step 4: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 1000)
Y_Train = Y_Train.reshape(-1,1)
X_Train = X_Train.reshape(-1,1)
X_Test = X_Test.reshape(-1,1)

# Step 5: Fitting the SVR model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_Train,Y_Train)

# Step 6: Making prediction
Y_Pred = regressor.predict(X_Test)

# Step 7: Printing the RMSE
Y_Test = Y_Test.reshape(-1,1)
Y_Pred = Y_Pred.reshape(-1,1)
testScore = math.sqrt(mean_squared_error(Y_Test[:,0], Y_Pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Step 8: Plotting the results
plt.scatter(X_Test,Y_Test, color = 'red')
plt.scatter(X_Test, Y_Pred, color = 'blue')
plt.title('Regression Results')
plt.xlabel('Num of days')
plt.ylabel('Forex rate')
plt.show()