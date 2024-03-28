import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset

# Author: Mnelisi Mabuza
# Data science intern @CodeClause
# Forecasting future sales using ARIMA model


# Step 1: read in the data file
salesData = pd.read_csv('Stuttafords.csv')
# model = ARIMA(salesData, order(5,1,0))

# print(salesData.tail())
salesData.columns=["Month","Sales"]
# print(salesData.head())

# Step 2: Filtering and cleaning of unwanted data!

# Identify non-date values in the 'Month' column
non_date_values = salesData[~salesData['Month'].str.match(r'\d{4}-\d{2}', na=False)]['Month']
# Remove rows with non-date values
salesData = salesData[salesData['Month'].str.match(r'\d{4}-\d{2}', na=False)]
# salesData.drop(105,axis=0,inplace=True)
# Convert 'Month' to date-time format
try:
    salesData['Month'] = pd.to_datetime(salesData['Month'], format='%Y-%m')
except ValueError as e:
    print(f"Error: {e}")
    print("Non-date values found in the 'Month' column. Please clean the data before proceeding.")
    exit(1)
# print(salesData.tail())

salesData.set_index('Month', inplace=True)
# print(salesData.describe())
# salesData.plot()

# STEP 3 : CONDUCT A ADF TEST TO SEE IF DATA IS STATIONARY OR NOT

test = adfuller(salesData['Sales'])

# Hypothesis testing
# H0 It is not stationary
# H1 It is stationary

def adfullerTest(sales):
    r=adfuller(sales)
    variables = ['ADF Test statistic', 'p-value','#Lags Used', '# of Observations Used']
    for value, variable in zip(r, variables):
        print(variable+' : '+str(value))
    if r[1] <= 0.05:
        print("Evident to reject null hypothesis p value is less that 0.05")
    else:
        print("Weak evidence against null hypothesis, time series has unit root thus non stationary")
    
# adfullerTest(salesData['Sales'])
# Differencing of observations per year[12] months to make the data stationary

salesData['Sales_diff'] = salesData['Sales'] - salesData['Sales'].shift(12)
salesData.dropna(inplace=True)  

# Drop the first row with NaN

# Perform ADF test on the differenced data
adfullerTest(salesData['Sales_diff'])

# Lets visualize the differenced data set
#salesData['Sales_diff'].plot()    
#plt.show()

# Autocorrelation & Partial autocorrelation

graph = plt.figure(figsize=(12,8))
axisOne = graph.add_subplot(211)
graph = sm.graphics.tsa.plot_acf(salesData['Sales_diff'].iloc[13:],lags=40,ax=axisOne)
axis2 = graph.add_subplot(212)
graph = sm.graphics.tsa.plot_pacf(salesData['Sales_diff'].iloc[13:],lags=40,ax=axis2)

model=ARIMA(salesData['Sales'],order=(1,1,1))
model_fit=model.fit()

#print(model_fit.summary())

#model=sm.tsa.statespace.SARIMAX(salesData['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
#results=model.fit()
# Fit SARIMA model
model = sm.tsa.statespace.SARIMAX(salesData['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

#salesData['forecast']=results.predict(start=90,end=106,dynamic=True)
#salesData[['Sales','forecast']].plot(figsize=(12,8))

# Plot the forecast

forecasted=[salesData.index[-1]+ DateOffset(months=x)for x in range(0,24)]
fD=pd.DataFrame(index=forecasted[1:],columns=salesData.columns)
futureDf=pd.concat([salesData,fD])
futureDf['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))

plt.show()


