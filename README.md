# Deep Learning Project

## LOADING LIBRARIES

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```

## Load the dataset

```python
df = pd.read_csv("AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.head()
```

## Visualize the data

```python
plt.figure(figsize=(10, 5))
plt.plot(df, label='Passengers')
plt.title('Monthly Air Passengers')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
```

## NORMALIZE THE DATA

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['#Passengers']])
```

## PREPARE DATA FOR LSTM MODEL

```python
def create_sequences(data, time_steps=12):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_steps = 12
X, y = create_sequences(scaled_data, time_steps)

# Reshape X for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
```

## MODEL BUILDING

```python
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
```

## TRAIN THE MODEL

```python
model.fit(X, y, epochs=100, batch_size=16, verbose=1)
```

## FORECASTING

## Start with the last time_steps from the training data
last_sequence = scaled_data[-time_steps:]
forecast = []

current_input = last_sequence.reshape(1, time_steps, 1)

for _ in range(12):  # predict next 12 months
    next_val = model.predict(current_input, verbose=0)
    forecast.append(next_val[0, 0])
    
    # update the input sequence
    next_val_reshaped = next_val.reshape(1, 1, 1)  # Make it 3D: (1, 1, 1)
    current_input = np.append(current_input[:, 1:, :], next_val_reshaped, axis=1)


# Inverse transform to get actual values
forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

## PLOT

## Prepare forecast timeline
forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['#Passengers'], label='Actual')
plt.plot(forecast_dates, forecast_actual, color='red', label='Forecast')
plt.title('Air Passengers Forecast using LSTM')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()

```python

```



