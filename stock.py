#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
plt.style.use('fivethirtyeight')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[8]:


data_dir = 'tesla-stock-price.csv'

# parse_dates make date if like this 12/10/2010 to be like that 12-10-2010
# index_col to make col 'Date' is index

df = pd.read_csv(data_dir, index_col='date', parse_dates=True) 


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[13]:


plt.figure(figsize=(15,6))
df['open'].plot()
df['close'].plot()
plt.xlabel(None)
plt.ylabel(None)
plt.title('Opening & Closing Price of TESLA')
plt.legend(['Open Price', 'Close Price'])
plt.tight_layout()
plt.show()


# In[16]:


dataset = df['close']
dataset = pd.DataFrame(dataset)

data = dataset.values
data.shape


# In[17]:


#normalizing the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# In[18]:


scaled_data


# In[19]:


train_size = int(len(data) * .75)     # 2217
test_size = len(data) - train_size    # 739      # same code int(len(data) * .25)   

print('Train Size : ', train_size, 'Test Size : ', test_size)

train_data = scaled_data[:train_size, 0:1]
test_data = scaled_data[train_size-60:, 0:1]


# In[20]:


print('Shape of Train Data : ', train_data.shape)
print('Shape of Test Data', test_data.shape)


# In[21]:


# Creating a Training Set with 60 time-steps and 1 output
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60 : i , 0])
    y_train.append(train_data[i , 0])


# In[22]:


# Convert to numpy array 
x_train, y_train = np.array(x_train) , np.array(y_train)
# Reshaping the input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train
x_train.shape, y_train.shape


# In[23]:


model = Sequential([
    LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)),
    LSTM(64, return_sequences=False),
    
    Dense(32),
    Dense(16),
    Dense(1)
])

model.compile(optimizer='adam', loss= 'mse', metrics=['mean_absolute_error'])


# In[24]:


model.summary()


# In[25]:


early_stopping = EarlyStopping(
    patience=10,
    monitor='loss',
    restore_best_weights=True
)


# In[26]:


# Fitting the LSTM to the Training set
history = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=early_stopping)


# In[27]:


plt.figure(figsize=(15,6))
plt.plot(history.history['loss'])
plt.plot(history.history['mean_absolute_error'])
plt.legend(['Mean Squared Error', 'Mean Absolute Error'])
plt.title('Losses')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


# In[28]:


# Creating a Testing Set with 60 time-steps and 1 output
x_test = []
y_test = []

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i,0])


# In[29]:


x_test , y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape, y_test.shape


# In[30]:


# inverse y_test scaling 
predictions = model.predict(x_test)

# inverse predictions scaling 
predictions = scaler.inverse_transform(predictions)
predictions.shape


# In[31]:


# inverse y_test scaling 
y_test = scaler.inverse_transform([y_test])

RMSE = np.sqrt(np.mean(y_test - predictions)**2).round(2)
RMSE


# In[34]:


from datetime import timedelta


# In[35]:


def insert_end(Xin, new_input):
    timestep = 60 
    for i in range(timestep - 1):    # This loop for making shift to build new Xin
        Xin[:, i, :] = Xin[:, i+1, :]
    Xin[:, timestep-1, :] = new_input
    return Xin


# In[36]:


future = 30 
forecast = []
Xin = x_test[-1: , :, :]
time = []

for i in range(0, future):
    out = model.predict(Xin, batch_size=5)
    forecast.append(out[0,0])
    print(forecast)
    Xin = insert_end(Xin, out[0,0])
    time.append(pd.to_datetime(df.index[-1]) + timedelta(days=i))


# In[37]:


time


# In[38]:


forecast


# In[39]:


forecasted_output = np.array(forecast)
forecasted_output = forecasted_output.reshape(-1,1)
forecasted_output = scaler.inverse_transform(forecasted_output)
forecasted_output = pd.DataFrame(forecasted_output)
date = pd.DataFrame(time)
df_result = pd.concat([date, forecasted_output], axis=1)
df_result.columns = 'Date', 'Forecasted'


# In[40]:


df_result


# In[41]:


df_result.set_index('Date', inplace=True)
df_result


# In[43]:


plt.figure(figsize=(15,7))
df['close'].plot(linewidth=3)
df_result['Forecasted'].plot(linewidth=3)
plt.title('TESLA Close Stock Price Forecasting For Next 30 Days')
plt.xlabel('Date', fontsize = 17)
plt.ylabel('Close', fontsize = 17)
plt.show()


# In[ ]:




