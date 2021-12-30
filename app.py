### Import/Data Collections
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import streamlit as st1

### Getting data from Yahoo finance
start = '2010-01-01'
end = '2021-12-01'

st.title('Stock Trend Predictions')

userInput = st.text_input('Please Enter a Stock Ticker', 'AAPL')
df= pdr.DataReader(userInput,'yahoo', start, end)



st.subheader('Data ranging')
st.write(df.describe())

## Plotting 
st.subheader('Closing Price  vs Time Chart')
fig=plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price  vs 100 Days Moving Average')
ma100 = df.Close.rolling(100).mean()
fig=plt.figure(figsize= (12,6))
plt.plot(ma100,'r',label='Moving Average for 100 Days')
plt.legend()
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price  vs Time with 100 and 200 Days Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig=plt.figure(figsize= (12,6))
plt.plot(ma100,'r',label='Moving Average for 100 Days')
plt.plot(ma200,'g',label='Moving Average for 200 Days')
plt.legend()
plt.plot(df.Close)
st.pyplot(fig)


## preprocessing data for model
df1=df['Close']
## selecting only Close colummn
df1=df.reset_index()['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
##splitting dataset into training and testing
training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix 
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

##Loading thge model
model = load_model('keras_model200.h5')

X_predict=model.predict(X_train)
Y_predict=model.predict(X_test)


X_predict=scaler.inverse_transform(X_predict)
Y_predict=scaler.inverse_transform(Y_predict)


scale_factor = 1/1.
y_predicted = Y_predict * scale_factor
x_predicted = X_predict * scale_factor

# plot baseline and predictions
st.subheader('Trained and Test Model Predictiont')
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(X_predict)+look_back, :] = X_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(X_predict)+(look_back*2)+1:len(df1)-1, :] = Y_predict
# plot baseline and predictions
fig2=plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1),label='Original Price')
#plt.plot(scaler.inverse_transform(df2),label='Original Price')
plt.plot(trainPredictPlot,label=' Trained Price')
plt.plot(testPredictPlot,label=' Test Price')
plt.legend()
plt.show()
st.pyplot(fig2)

##New 
x_input=test_data[len(test_data)-100:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
day_new=np.arange(1,101)
day_pred=np.arange(101,131)

st.subheader('Predicting next 30 days prices')
fig3=plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-100:]),label='100 Days Value')
plt.plot(day_pred,scaler.inverse_transform(lst_output),label ='Next 30 Days Prediction')
plt.legend()
st.pyplot(fig3)
#print(scaler.inverse_transform(lst_output))




