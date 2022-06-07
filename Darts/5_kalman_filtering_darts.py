# Source: https://github.com/unit8co/darts/blob/master/examples/10-Kalman-filter-examples.ipynb

from turtle import clear
from utils import *
fix_pythonpath_if_working_locally()

import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import KalmanFilter
from darts.utils import timeseries_generation as tg

clear_terminal()

NOISE_DISTANCE = 0.1
SAMPLE_SIZE = 200
np.random.seed(42)

# Prepare the input
u = TimeSeries.from_values(np.heaviside(np.linspace(-5, 10, SAMPLE_SIZE), 0))

# Prepare the output
y = u * TimeSeries.from_values(1 - np.exp(-np.linspace(-5, 10, SAMPLE_SIZE)))

# Add white noise to obtain the observations
noise = tg.gaussian_timeseries(length=SAMPLE_SIZE, std=NOISE_DISTANCE)
y_noise = y + noise

plt.figure(figsize=[12, 8])
u.plot(label="Input")
y.plot(color="gray", label="Output")
y_noise.plot(color="red", label="Noisy observations")
plt.legend()

# Traning
kf = KalmanFilter(dim_x=1)
kf.fit(y_noise, u)

# Filtering
y_filtered = kf.filter(y_noise, u)

plt.figure(figsize=[12, 8])
u.plot(label="Input")
y.plot(color="gray", label="Output")
y_noise.plot(color="red", label="Noisy observations")
y_filtered.plot(color="blue", label="Filtered observations")
plt.legend()



# Probababilistic Estimate
y_filtered = kf.filter(y_noise, u, num_samples=1000)

plt.figure(figsize=[12, 8])
u.plot(label="Input")
y.plot(color="gray", label="Output")
y_noise.plot(color="red", label="Noisy observations")
y_filtered.plot(color="blue", label="Filtered observations")
plt.legend()

plt.show()

