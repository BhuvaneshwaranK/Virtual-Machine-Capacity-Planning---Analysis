#Importing all required libraries for forecasting
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf



#Importing data
#In this analysis, Forecasting is done using only 'pageout' metrics. For other metrics, change the data page here.
df = pd.read_excel('Final.xlsx',sheetname='PAGEOUT')
df_needed = df[['rpttime','value']]

#Creating train and test set 
#Taking 70% for train and 30% for test set.
train=df_needed[0:int(0.7*len(df_needed))] 
test=df_needed[int(0.7*len(df_needed)):]

#Converting rpptime to datetime format
df_needed.rpttime = pd.to_datetime(df_needed.rpttime,format='%d-%m-%Y %H:%M')
train.rpttime = pd.to_datetime(train.rpttime,format='%d-%m-%Y %H:%M')
test.rpttime = pd.to_datetime(test.rpttime,format='%d-%m-%Y %H:%M')

#Eliminating Trend by taking log of time series data
ts = df_needed['value']
ts_log = np.log(ts)

#Differencing for treating any Trend and Seasonality
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)


#Basic Naiva approach
def naive_approach(train,test):
    dd= np.asarray(train.value)
    y_hat = test.copy()
    y_hat['naive'] = dd[len(dd)-1]
    #To see forecasted values, print(y_hat.naive)
    #RMSE score
    rms = sqrt(mean_squared_error(test.value, y_hat.naive))
    return rms

#Simle Average method
def simple_average(train,test):
    y_hat_avg = test.copy()
    y_hat_avg['avg_forecast'] = train['value'].mean()
    #To see forecasted values, print(y_hat_avg.avg_forecast)
    #RMSE Score
    rms = sqrt(mean_squared_error(test.value, y_hat_avg.avg_forecast))
    return rms

#Moving Average method
def moving_average(train,test):
    y_hat_avg = test.copy()
    y_hat_avg['moving_avg_forecast'] = train['value'].rolling(60).mean().iloc[-1]
    #To see forecasted values, print(y_hat_avg.moving_avg_forecast)
    #RMSE Score
    rms = sqrt(mean_squared_error(test.value, y_hat_avg.moving_avg_forecast))
    return rms

#Autoregression (AR) Model 
def auto_regression(ts_log,ts_log_diff):
    model = ARIMA(ts_log, order=(1, 1, 0))  #Change 1 to some other values for better results
    results_AR = model.fit(disp=-1) 
    #Sum of squared errors (SSE) score
    sse = sum((results_AR.fittedvalues-ts_log_diff)**2) 
    #To see forecasted values, print(results_AR.fittedvalues)
    return sse

#ARIMA (Autoregression Integrated Moving average) model
def arima(ts_log,ts_log_diff):
    model = ARIMA(ts_log, order=(1, 1, 1))  #Change 1 to other values for better results
    results_ARIMA = model.fit(disp=-1) 
    #Sum of squared errors (SSE) score
    sse = sum((results_ARIMA.fittedvalues-ts_log_diff)**2) 
    #To see forecasted values, print(results_ARIMA.fittedvalues)
    return sse

naive_approach_rmse = naive_approach(train,test)
simple_average_rmse = simple_average(train,test)
moving_average_rmse = moving_average(train,test)
auto_regression_sse = auto_regression(ts_log,ts_log_diff)
arime_sse = arima(ts_log,ts_log_diff)

#Out of below models result, take the model with less RSME or SSE error
print("Naive approach RMSE == ",naive_approach_rmse)
print("Simple Average RMSE == ",simple_average_rmse)
print("Moving Average RMSE == ",moving_average_rmse)
print("Autoregression SSE == ",auto_regression_sse)
print("ARIME SSE == ",arime_sse)




