#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\imreh\Desktop\Feynn Internship\Shoe-Sales.csv")


# In[3]:


df.head()


# In[4]:


date = pd.date_range(start='1/1/2000', end='8/1/2015', freq='M')
date


# In[5]:


# Adding the time stamp to the data frame
df['Time_Stamp'] = pd.DataFrame(date)
df.head()


# In[6]:


df.set_index('Time_Stamp',inplace=True)
df.head()


# In[7]:


df.tail()


# In[8]:


# The following code is to set the subsequent figure sizes

from pylab import rcParams
rcParams['figure.figsize'] = 20,8


# In[9]:


df.plot()
plt.grid();


# In[10]:


df.describe()


# In[11]:


sns.boxplot(x = df.index.year,y = df['Shoe_Sales'])
plt.grid();


# In[12]:


sns.boxplot(x = df.index.month_name(),y = df['Shoe_Sales'])
plt.grid();


# In[13]:


from statsmodels.graphics.tsaplots import month_plot

month_plot(df['Shoe_Sales'],ylabel='Sales')
plt.grid()


# In[14]:


monthly_sales_across_years = pd.pivot_table(df, values = 'Shoe_Sales', columns = df.index.month, index = df.index.year)
monthly_sales_across_years


# In[15]:


monthly_sales_across_years.plot()
plt.grid()
plt.legend(loc='best');


# In[16]:


# statistics
from statsmodels.distributions.empirical_distribution import ECDF

plt.figure(figsize = (18, 8))
cdf = ECDF(df['Shoe_Sales'])
plt.plot(cdf.x, cdf.y, label = "statmodels");
plt.grid()
plt.xlabel('Sales');


# In[17]:


# group by date and get average RetailSales, and precent change
average    = df.groupby(df.index)["Shoe_Sales"].mean()
pct_change = df.groupby(df.index)["Shoe_Sales"].sum().pct_change()

fig, (axis1,axis2) = plt.subplots(2,1,sharex=True,figsize=(15,8))

# plot average RetailSales over time(year-month)
ax1 = average.plot(legend=True,ax=axis1,marker='o',title="Average RetailSales",grid=True)
ax1.set_xticks(range(len(average)))
ax1.set_xticklabels(average.index.tolist())
# plot precent change for RetailSales over time(year-month)
ax2 = pct_change.plot(legend=True,ax=axis2,marker='o',colormap="summer",title="RetailSales Percent Change",grid=True)


# In[20]:


from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss


# In[21]:


decomposition = seasonal_decompose(df['Shoe_Sales'],model='additive')
decomposition.plot();


# In[22]:


trend = decomposition.trend
seasonality = decomposition.seasonal
residual = decomposition.resid

print('Trend','\n',trend.head(12),'\n')
print('Seasonality','\n',seasonality.head(12),'\n')
print('Residual','\n',residual.head(12),'\n')


# In[23]:


decomposition = seasonal_decompose(df['Shoe_Sales'],model='multiplicative')
decomposition.plot();


# In[24]:


trend = decomposition.trend
seasonality = decomposition.seasonal
residual = decomposition.resid

print('Trend','\n',trend.head(12),'\n')
print('Seasonality','\n',seasonality.head(12),'\n')
print('Residual','\n',residual.head(12),'\n')


# In[25]:


## Test for stationarity of the series - Dicky Fuller test

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=7).mean() #determining the rolling mean
    rolstd = timeseries.rolling(window=7).std()   #determining the rolling standard deviation

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput,'\n')


# In[26]:


test_stationarity(df['Shoe_Sales'])


# In[27]:


test_stationarity(df['Shoe_Sales'].diff().dropna())


# In[31]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['Shoe_Sales'],lags=50)
plot_acf(df['Shoe_Sales'].diff().dropna(),lags=50,title='Differenced Data Autocorrelation')
plt.show()


# In[32]:


plot_pacf(df['Shoe_Sales'],lags=50)
plot_pacf(df['Shoe_Sales'].diff().dropna(),lags=50,title='Differenced Data Partial Autocorrelation')
plt.show()


# In[33]:


train=df[df.index.year < 2010]
test=df[df.index.year >= 2015]


# In[34]:


print('First few rows of Training Data')
display(train.head())
print('Last few rows of Training Data')
display(train.tail())
print('First few rows of Test Data')
display(test.head())
print('Last few rows of Test Data')
display(test.tail())


# In[35]:


print(train.shape)
print(test.shape)


# In[36]:


test_stationarity(train['Shoe_Sales'])


# In[37]:


test_stationarity(train['Shoe_Sales'].diff().dropna())


# In[38]:


train.info()


# In[39]:


## The following loop helps us in getting a combination of different parameters of p and q in the range of 0 and 2
## We have kept the value of d as 1 as we need to take a difference of the series to make it stationary.

import itertools
p = q = range(0, 3)
d= range(1,2)
pdq = list(itertools.product(p, d, q))
print('Some parameter combinations for the Model...')
for i in range(1,len(pdq)):
    print('Model: {}'.format(pdq[i]))


# In[40]:


# Creating an empty Dataframe with column names only
ARIMA_AIC = pd.DataFrame(columns=['param', 'AIC'])
ARIMA_AIC


# In[41]:


## Sort the above AIC values in the ascending order to get the parameters for the minimum AIC value

ARIMA_AIC.sort_values(by='AIC',ascending=True)

