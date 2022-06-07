# Source: https://github.com/unit8co/darts/blob/master/examples/01-multi-time-series-and-covariates.ipynb

from utils import *
fix_pythonpath_if_working_locally()

import numpy as np
import torch
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

from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)

series_air = AirPassengersDataset().load()
series_milk = MonthlyMilkDataset().load()

series_air.plot(label="Number of air passengers")
series_milk.plot(label="Pounds of milk produced per cow")
plt.legend()

# Preprocessing
scaler_air, scaler_milk = Scaler(), Scaler()
series_air_scaled = scaler_air.fit_transform(series_air)
series_milk_scaled = scaler_milk.fit_transform(series_milk)

series_air_scaled.plot(label="air")
series_milk_scaled.plot(label="milk")
plt.legend()

# Train / Validation split
# Let's keep the last 36 months of both series as validation:

train_air, val_air = series_air_scaled[:-36], series_air_scaled[-36:]
train_milk, val_milk = series_milk_scaled[:-36], series_milk_scaled[-36:]

# Example with One Series
# Let's look at a first example. We'll build an N-BEATS model that has a lookback 
# window of 24 points (input_chunk_length=24) and predicts the next 12 points 
# (output_chunk_length=12). We chose these values so it'll make our model produce 
# successive predictions for one year at a time, looking at the past two years.

n_epochs = 1    # change this
model_air = NBEATSModel(
    input_chunk_length=24, output_chunk_length=12, n_epochs=n_epochs, random_state=0
)

# Train the model
model_air.fit(train_air, verbose=True)

#  like any other Darts forecasting models, we can then get a forecast by calling predict(). 
# Note that below, we are calling predict() with a horizon of 36, which is longer than the 
# model internal output_chunk_length of 12. That's not a problem here - as explained above, 
# in such a case the internal model will simply be called auto-regressively on its own outputs. 
# In this case, it will be called three times so that the three 12-points outputs make up the 
# final 36-points forecast - but all of this is done transparently behind the scenes.

# Make predictions
pred = model_air.predict(n=36)

clear_terminal()

series_air_scaled.plot(label="actual")
pred.plot(label="forecast")
plt.legend()
print("MAPE = {:.2f}%".format(mape(series_air_scaled, pred)))

## Training on Both Air Traffic and Milk Series
# Let's look at another example where we fit another model instance on our two time series 
# (air passengers and milk production). Since using two series of (roughly) the same length 
# (roughly) doubles the training dataset size, we will use half of the number of epochs:

model_air_milk = NBEATSModel(
    input_chunk_length=24, output_chunk_length=12, n_epochs=100, random_state=0
)
# Then, fitting the model on two (or more) series is as simple as giving a list of series 
# (instead of a single series) in argument to the fit() function:

model_air_milk.fit([train_air, train_milk], verbose=True)


plt.close("all")
clear_terminal()

# Producing Forecasts After the End of a Series

pred = model_air_milk.predict(n=36, series=train_air)

series_air_scaled.plot(label="actual")
pred.plot(label="forecast")
plt.legend()
print("MAPE = {:.2f}%".format(mape(series_air_scaled, pred)))


plt.show()