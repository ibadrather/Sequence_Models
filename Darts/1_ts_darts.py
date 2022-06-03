"""
    Source: https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878
"""

import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing
import matplotlib.pyplot as plt

import os

# CLear terminal
try:
    os.system("clear")
except:
    pass

# Load Dataframe
df = pd.read_csv('AirPassengers.csv')
series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')

# In the above snippet, we first read a DataFrame containing the air passengers dataset. 
# We then build a (univariate) TimeSeries, specifying the time and value columns 
# (Month and #Passengers, respectively).


# Let’s now split our series in a training and validation TimeSeries, 
# and train an exponential smoothing model on the training series:

train, val = series.split_before(pd.Timestamp('19580101'))
model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=500)

series.plot(label='actual')
prediction.plot(label='ExponentialSmoothing forecast', lw=3)
plt.legend()
plt.show()

# Auto-ARIMA Model for the same
from darts.models import AutoARIMA

model_aarima = AutoARIMA()
model_aarima.fit(train)
prediction_aarima = model_aarima.predict(len(val))

series.plot(label='actual')
prediction_aarima.plot(label='AutoARIMA forecast', lw=3)
plt.legend()
plt.show()

