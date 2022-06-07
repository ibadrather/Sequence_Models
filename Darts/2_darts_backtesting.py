"""
    Source: https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878
"""
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing, Prophet, AutoARIMA
import matplotlib.pyplot as plt

from darts.metrics import mape

import os

# CLear terminal
try:
    os.system("clear")
except:
    pass

# Load Dataframe
df = pd.read_csv('AirPassengers.csv')
series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')

# In our example above, we used Darts to obtain once a forecast over the next 36 months starting 
# in January 1958. However, forecasts often need to be updated as soon as new data becomes available. 
# With Darts, it’s easy to compute the forecasts resulting from such a process, using backtesting. 
# For instance, using backtesting to compare two models looks as follows:

models = [ExponentialSmoothing(), AutoARIMA()] # , Prophet()
backtests = [model.historical_forecasts(series,
                            start=.5,
                            forecast_horizon=3)
             for model in models]


series.plot(label='data')
for i, m in enumerate(models):
    err = mape(backtests[i], series)
    backtests[i].plot(lw=3, label='{}, MAPE={:.2f}%'.format(m, err))
plt.title('Backtests with 3-months forecast horizon')
plt.legend()
plt.show()