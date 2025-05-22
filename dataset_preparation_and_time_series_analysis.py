import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import time
import json
import glob

files = glob.glob(os.path.join('/content/drive/MyDrive/targetdir/rnd/2013-7', "*.csv"))
files_first200 = files[:150]
dfs = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files_first200]
df = pd.concat(dfs, ignore_index=True)

files2 = glob.glob(os.path.join('/content/drive/My Drive/targetdir/rnd/2013-8', "*.csv"))
files2_first200 = files2[:150]
dfs2 = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files2_first200]
df2 = pd.concat(dfs2, ignore_index=True)

files3 = glob.glob(os.path.join('/content/drive/My Drive/targetdir/rnd/2013-9', "*.csv"))
files3_first200 = files3[:150]
dfs3 = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files3_first200]
df3 = pd.concat(dfs3, ignore_index=True)

concatenated_df = pd.concat([df, df2, df3])
concatenated_df.head()

"""# **Feature Engineering**"""

concatenated_df['Timestamp'] = pd.to_datetime(concatenated_df['Timestamp [ms]'], unit = 's')
concatenated_df.apply(pd.to_numeric, errors='ignore')

# Date Feature Engineering
concatenated_df['weekday'] = concatenated_df['Timestamp'].dt.dayofweek
concatenated_df['weekend'] = ((concatenated_df.weekday) // 5 == 1).astype(float)
concatenated_df['month']=concatenated_df.Timestamp.dt.month
concatenated_df['day']=concatenated_df.Timestamp.dt.day
concatenated_df.set_index('Timestamp',inplace=True)

concatenated_df["CPU usage prev"] = concatenated_df['CPU usage [%]'].shift(1)
concatenated_df["CPU_diff"] = concatenated_df['CPU usage [%]'] - concatenated_df["CPU usage prev"]
concatenated_df["received_prev"] = concatenated_df['Network received throughput [KB/s]'].shift(1)
concatenated_df["received_diff"] = concatenated_df['Network received throughput [KB/s]']- concatenated_df["received_prev"]
concatenated_df["transmitted_prev"] = concatenated_df['Network transmitted throughput [KB/s]'].shift(1)
concatenated_df["transmitted_diff"] = concatenated_df['Network transmitted throughput [KB/s]']- concatenated_df["transmitted_prev"]
concatenated_df = concatenated_df.fillna(method='ffill')
concatenated_df.head()
hourlydat = concatenated_df.resample('H').sum()
hourlydat.head()

"""# **Throught Put Graph**"""

import pandas as pd
import matplotlib.pyplot as plt

df_hourly_transmitted = hourlydat['Network transmitted throughput [KB/s]'].resample('H').max()
df_hourly_received = hourlydat['Network received throughput [KB/s]'].resample('H').max()

# Plotting the Network Transmitted Throughput (Hourly Max)
plt.figure(figsize=(12, 6))
plt.plot(df_hourly_transmitted.index, df_hourly_transmitted, label='Network Transmitted Throughput (Max)', color='blue')
plt.title('Network Transmitted Throughput (Hourly Max)')
plt.xlabel('Time')
plt.ylabel('Throughput')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the Network Received Throughput (Hourly Max)
plt.figure(figsize=(12, 6))
plt.plot(df_hourly_received.index, df_hourly_received, label='Network Received Throughput (Max)', color='green')
plt.title('Network Received Throughput (Hourly Max)')
plt.xlabel('Time')
plt.ylabel('Throughput')
plt.legend()
plt.grid(True)
plt.show()

"""# **Cpu usage vs Capacity provisioned**"""

import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(hourlydat.index, hourlydat['CPU usage [MHZ]'], label='CPU Usage', color='green')
plt.plot(hourlydat.index, hourlydat['CPU capacity provisioned [MHZ]'], label='Capacity Provisioned', color='red')
plt.title('CPU Usage vs. Capacity Provisioned Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

df['CPU usage [MHZ]'].max()
df['CPU capacity provisioned [MHZ]'].max()

"""# **Time series Analysis test**"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

window_size = 24
df['moving_average'] = df['CPU usage [MHZ]'].rolling(window=window_size).mean()
df['difference'] = df['CPU usage [MHZ]'] - df['moving_average']
std_dev = np.std(df['difference'].dropna())

threshold = 2 * std_dev
anomalies = df[abs(df['difference']) > threshold]
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['CPU usage [MHZ]'], label='Actual CPU Usage', color='blue')
plt.plot(df.index, df['moving_average'], label='Moving Average (24-hour window)', color='orange')

plt.scatter(anomalies.index, anomalies['CPU usage [MHZ]'], color='red', label='Anomalies', marker='o')
plt.title('CPU Usage with Moving Average and Anomalies')
plt.xlabel('Time')
plt.ylabel('CPU Usage')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/targetdir/rnd/df_scaled.csv")
plt.plot(df['CPU usage [MHZ]'])
plt.title("Time Series Data")
plt.show()

"""## **ADF and KPSS**"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

# ADF Test
adf_result = adfuller(hourlydat['CPU usage [MHZ]'].dropna())
print("ADF Test")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print(f"Critical Values:")
for key, value in adf_result[4].items():
    print(f'   {key}: {value}')

# KPSS Test
kpss_result = kpss(hourlydat['CPU usage [MHZ]'].dropna(), regression='c')
print("\nKPSS Test")
print(f"KPSS Statistic: {kpss_result[0]}")
print(f"p-value: {kpss_result[1]}")
print(f"Critical Values:")
for key, value in kpss_result[3].items():
    print(f'   {key}: {value}')

"""## **shapiro wilko test**"""

import pandas as pd
from scipy.stats import shapiro
stat, p_value = shapiro(hourlydat['CPU usage [MHZ]'].dropna())

print("Shapiro-Wilk Test")
print(f"Test Statistic: {stat}")
print(f"p-value: {p_value}")

"""# Detrending the data"""

import numpy as np
from sklearn.linear_model import LinearRegression

time = np.arange(len(df['CPU usage [MHZ]'])).reshape(-1, 1)
model = LinearRegression()
model.fit(time, df['CPU usage [MHZ]'])
trend = model.predict(time)
detrended = df['CPU usage [MHZ]'] - trend
plt.plot(detrended)
plt.title("Detrended Data")
plt.show()


from statsmodels.tsa.stattools import adfuller
result = adfuller(detrended)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] < 0.05:
    print("The detrended series is stationary (Reject H0)")
else:
    print("The detrended series is not stationary (Fail to Reject H0)")

from statsmodels.tsa.stattools import kpss

result_kpss = kpss(detrended, regression='c')
print('KPSS Statistic:', result_kpss[0])
print('p-value:', result_kpss[1])
if result_kpss[1] > 0.05:
    print("The detrended series is stationary (Fail to Reject H0)")
else:
    print("The detrended series is not stationary (Reject H0)")

"""# **Seasonal and trend analysis**"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(hourlydat['CPU usage [MHZ]'].dropna(), model='additive', period=24)

plt.figure(figsize=(12, 10))
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual', color='red')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

"""# **Data Preparation for AWS Forecast**"""

concatenated_df.head()
concatenated_df['Timestamp'] = pd.to_datetime(concatenated_df['Timestamp [ms]'], unit = 's')
concatenated_df.describe()
concatenated_df['weekday'] = concatenated_df['Timestamp'].dt.dayofweek
concatenated_df['weekend'] = ((concatenated_df.weekday) // 5 == 1).astype(float)

concatenated_df['month']=concatenated_df.Timestamp.dt.month
concatenated_df['day']=concatenated_df.Timestamp.dt.day
concatenated_df.set_index('Timestamp',inplace=True)
concatenated_df["CPU usage prev"] = concatenated_df['CPU usage [%]'].shift(1)
concatenated_df["CPU_diff"] = concatenated_df['CPU usage [%]'] - concatenated_df["CPU usage prev"]
concatenated_df["received_prev"] = concatenated_df['Network received throughput [KB/s]'].shift(1)
concatenated_df["received_diff"] = concatenated_df['Network received throughput [KB/s]']- concatenated_df["received_prev"]
concatenated_df["transmitted_prev"] = concatenated_df['Network transmitted throughput [KB/s]'].shift(1)
concatenated_df["transmitted_diff"] = concatenated_df['Network transmitted throughput [KB/s]']- concatenated_df["transmitted_prev"]

concatenated_df["start"] = concatenated_df.index
concatenated_df['target'] = concatenated_df['CPU usage [MHZ]']

import pandas as pd

concatenated_df = pd.read_csv("/content/drive/MyDrive/targetdir/rnd/concatenated_df.csv")
concatenated_df.head()

"""**Resampling the data to minute intervals**"""

df = concatenated_df.groupby('VM').resample('1min')['target'].mean().to_frame()
df.reset_index(level=0, inplace=True)
df = df2.fillna(method='ffill')
df = df.fillna(method='ffill')
df.head()
df.rename(columns={"Timestamp": "timestamp", "VM": "item_id", "target": "target_value"}, inplace=True)
df.rename_axis("timestamp", inplace=True)
df.head(20)
df.to_csv("preprocessed_data.csv")

"""# **Visualization from the results of AWS:**"""

import matplotlib.pyplot as plt

models = ['ARIMA', 'Proposed DeepAR+', 'Proposed CNN-QR']
mape_values = [29.61, 13.00, 6.48]
plt.figure(figsize=(10, 6))
plt.bar(models, mape_values, color=['#FFB6C1', '#ADD8E6', '#98FB98'])  # Light Pink, Light Blue, Light Green
plt.xlabel('Model')
plt.ylabel('MAPE (%)')
plt.title('MAPE Comparison: ARIMA vs Proposed DeepAR+ vs Proposed CNN-QR')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('arima_vs_proposed.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt

models = ['LSTM', 'Proposed DeepAR+', 'Proposed CNN-QR']
mape_values = [69.50, 13.00, 6.48]
plt.figure(figsize=(10, 6))
plt.bar(models, mape_values, color=['#FFB6C1', '#ADD8E6', '#98FB98'])  # Light Pink, Light Blue, Light Green
plt.xlabel('Model')
plt.ylabel('MAPE (%)')
plt.title('MAPE Comparison: LSTM vs Proposed DeepAR+ vs Proposed CNN-QR')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('lstm_vs_proposed_light_colors.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt

models = ['DeepAR (CloudAIBus)', 'Proposed DeepAR+', 'Proposed CNN-QR']
mape_values = [16.75, 13.00, 6.48]
plt.figure(figsize=(10, 6))
plt.bar(models, mape_values, color=['#FFB6C1', '#ADD8E6', '#98FB98'])  # Light Pink, Light Blue, Light Green
plt.xlabel('Model')
plt.ylabel('MAPE (%)')
plt.title('MAPE Comparison: DeepAR vs Proposed DeepAR+ vs Proposed CNN-QR')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('deepar_vs_proposed.png', dpi=300)
plt.show()