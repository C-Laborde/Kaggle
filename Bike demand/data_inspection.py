# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3.7.5 64-bit
#     language: python
#     name: python37564bit82d48e3e9c1b4c058e1f99865c7226e7
# ---

# #### Goal
# Use historical bike usage patterns with weather data to forecast bike rental demand

# #### Data description
# - datetime: hourly date + timestamp
# - season:</br>
#     1 = spring</br>
#     2 = summer</br>
#     3 = fall</br>
#     4 = winter
# - holiday: whether the day is considered a holiday
# - workingday: whether the day is neither a weekend nor holiday
# - weather:</br>
#     1: Clear, Few clouds, Partly cloudy, Partly cloudy </br>
#     2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist </br>
#     3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds </br>
#     4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp: temperature in Celsius
# - atemp: "feels like" temperature in Celsius
# - humidity: relative humidity
# - windspeed: wind speed
# - casual: number of non-registered user rentals initiated
# - registered: number of registered user rentals initiated
# - count: number of total rentals (Dependent Variable)
#

# import heatmapz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# %matplotlib inline

plot_dir = "plots/"

df = pd.read_csv("data/train/train.csv")

df.head(3)

df.describe()

# </br>
#
# ### Feature engineering

# **Datetime format**

df["datetime"] = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S")
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour

# </br>
#
# **Categorical variables**

# +
# Categorical variables:
categ_vars = ["season", "weather", "holiday", "workingday"]

for var in categ_vars:
    df[var] = df[var].astype("category")
# -

# <br/>
#
# **Missing values**

# Check for missing values (Obs: there are no missing values this time)
df.isnull().sum()

# </br>
#
# **Time series visualisation**

# OBS: The train data only contains the first 20 days of the month, that's why
# there are periodic gaps in the plots

# +
vars_to_plot = ["temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]
l = int(len(vars_to_plot) / 2) + 1 * len(vars_to_plot) % 2

# plt.rcParams["figure.figsize"] = [24, 12]
fig, ax = plt.subplots(l, 2, figsize=(28, 36))
sns.set(font_scale=3)

k = 0
for i in range(l):
    for j in range(2):
        try: 
            sns.lineplot(df.datetime, df[vars_to_plot[k]], ax=ax[i][j])
            ax[i][j].set_title(vars_to_plot[k])
            ax[i][j].tick_params(axis='x', labelrotation=45)
            k += 1
        except:
            break

fig.delaxes(ax[l-1,1])
plt.tight_layout()
filename = "variables_time_series.pdf"
if False:
    plt.savefig(plot_dir + filename)
# -

# </br>
#
# **Count on different seasons, months and hours**



# </br>
#
# **Correlation analysis** 

# - "temp" and "atemp" has a strong correlation and one of them should be dropped before building the model
# - "casual" and "registered" have some strong correlation with the "count" variable which seems reasonable considering that count = casual + registered

# Can't use Pearson correlation on categorical vars
vars_to_corr = ["temp", "atemp", "casual", "registered", "humidity"]
corr = df[vars_to_corr].corr()

# +
# mask top-right triangle
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False

sns.set(font_scale=1.2)
plt.rcParams["figure.figsize"] = [9, 9]
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(20, 220, n=200),
            vmin=-1, center=0, square=True, cbar_kws={"shrink": .82}, annot=True)
plt.yticks(rotation=0)

filename = "correlation_plot.pdf"
if True:
    plt.savefig(plot_dir + filename)

# TODO: how to avoid annotating the identity?
# -


