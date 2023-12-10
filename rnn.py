import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('HistoricalData.csv')  
prices = data['Close/Last'].values.reshape(-1, 1)  

scaler = MinMaxScaler(feature_range=(0, 1))
prices_normalized = scaler.fit_transform(prices)

look_back = 30 
X, Y = [], []
for i in range(len(prices_normalized) - look_back - 14):  
    X.append(prices_normalized[i:(i + look_back), 0])
    Y.append(prices_normalized[i + look_back:i + look_back + 14, 0])
X, Y = np.array(X), np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

batch_size = 64
optimizer = 'adam'
epochs = 50

model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=14))  
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

loss = model.evaluate(X_test, y_test)
print("Test Loss: ", loss)

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test[-1], label='Actual Prices')
plt.plot(predictions[-1], label='Predicted Prices')
plt.title('AAPL Stock Prices - Actual vs Predicted (2 Weeks)')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()


