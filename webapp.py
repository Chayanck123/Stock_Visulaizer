import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yahoo_fin.stock_info as si

start = "2010-01-01"
end = '2019-12-31'

st.title('Stock Trend Prediction')

# Get the list of available tickers
tickers = si.tickers_sp500()

# Allow the user to choose a ticker
ticker = st.selectbox('Select a ticker', tickers)

df = si.get_data(ticker, start_date=start, end_date=end)

#Describing Data
st.subheader('Data from 2010-2019')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize= (12,6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.close, 'b')
st.pyplot(fig)

#Splitting Data into Training and Testing sets
data_training = pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70): int(len(df))])
print(data_training.shape, data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)


#Load my model
model = load_model('keras_model.h5')

#Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)