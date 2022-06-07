# Source: https://github.com/unit8co/darts/blob/master/examples/01-multi-time-series-and-covariates.ipynb

import os
from utils import *

fix_pythonpath_if_working_locally()

import matplotlib.pyplot as plt

from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
)

from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset


# Covariates Series
# Until now, we have only been playing with models that only use the history of the target series to predict its future. 
# However, as explained above, the global Darts models also support the use of covariates time series. These are time 
# series of "external data", which we are not necessarily interested in predicting, but which we would still like to feed 
# as input of our models because they can contain valuable information.

# Loading the Series
series_air = AirPassengersDataset().load()
series_milk = MonthlyMilkDataset().load()

# Preprocessing
scaler_air, scaler_milk = Scaler(), Scaler()
series_air_scaled = scaler_air.fit_transform(series_air)
series_milk_scaled = scaler_milk.fit_transform(series_milk)


# Train / Validation split
# Let's keep the last 36 months of both series as validation:
train_air, val_air = series_air_scaled[:-36], series_air_scaled[-36:]
train_milk, val_milk = series_milk_scaled[:-36], series_milk_scaled[-36:]

# Building Covariates
# Let's see a simple example with our air and milk series, where we'll try to use the year and month-of-the-year as covariates:

# build year and month series:
air_year = datetime_attribute_timeseries(series_air_scaled, attribute="year")
air_month = datetime_attribute_timeseries(series_air_scaled, attribute="month")

milk_year = datetime_attribute_timeseries(series_milk_scaled, attribute="year")
milk_month = datetime_attribute_timeseries(series_milk_scaled, attribute="month")

# stack year and month to obtain series of 2 dimensions (year and month):
air_covariates = air_year.stack(air_month)
milk_covariates = milk_year.stack(milk_month)

# scale them between 0 and 1:
scaler_dt_air = Scaler()
air_covariates = scaler_dt_air.fit_transform(air_covariates)

scaler_dt_milk = Scaler()
milk_covariates = scaler_dt_milk.fit_transform(milk_covariates)

# split in train/validation sets:
air_train_covariates, air_val_covariates = air_covariates[:-36], air_covariates[-36:]
milk_train_covariates, milk_val_covariates = (
    milk_covariates[:-36],
    milk_covariates[-36:],
)

# plot the covariates:
plt.figure()
air_covariates.plot()
plt.title("Air traffic covariates (year and month)")

plt.figure()
milk_covariates.plot()
plt.title("Milk production covariates (year and month)")



plt.close("all")

# Training with Covariates
# Let's revisit our example again, this time with covariates. We will build a BlockRNNModel here:

model_cov = BlockRNNModel(
    model="LSTM",
    input_chunk_length=24,
    output_chunk_length=12,
    n_epochs=300,
    random_state=0,
)

# Training the model
model_cov.fit(
    series=[train_air, train_milk],
    past_covariates=[air_train_covariates, milk_train_covariates],
    verbose=True,
)

# Forecasting with Covariates
# similarly, getting a forecast is now only a matter of specifying the future_covariates argument to the predict() function.

pred_cov = model_cov.predict(n=36, series=train_air, past_covariates=air_covariates)

series_air_scaled.plot(label="actual")
pred_cov.plot(label="forecast")
plt.legend()


# Backtesting with Covariates
# We can also backtest the model using covariates. Say for instance we are interested in evaluating the running accuracy 
# with a horizon of 12 months, starting at 75% of the air series:

backtest_cov = model_cov.historical_forecasts(
    series_air_scaled,
    past_covariates=air_covariates,
    start=0.75,
    forecast_horizon=12,
    stride=1,
    retrain=False,
    verbose=True,
)

clear_terminal()

plt.close("all")

series_air_scaled.plot(label="actual")
backtest_cov.plot(label="forecast")
plt.legend()
print("MAPE (using covariates) = {:.2f}%".format(mape(series_air_scaled, backtest_cov)))

plt.show()