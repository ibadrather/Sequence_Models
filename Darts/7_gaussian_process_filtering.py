# Author: Ibad Rather, ibad.rather.ir@gmail.com, June 2022
# https://github.com/unit8co/darts/blob/master/examples/11-GP-filter-examples.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from utils import clear_terminal
from utils import *

from darts import TimeSeries

clear_terminal()

def add_white_noise(signal):
    sigma = 0.09
    return np.random.normal(signal, sigma, signal.shape[0])

# Generate timestamps
signal_length = 30
signal_frequency = 30 # hz
timestamps = np.linspace(0, signal_length, signal_length*signal_frequency)

# Generate Sin wave
omega = 2
signal = np.sin(omega*timestamps)

# Adding Noise to it
noisy_signal = add_white_noise(signal)
plt.plot(timestamps, signal, color="r", label="Signal")
plt.plot(timestamps, noisy_signal, color="b", label="Noisy Signal")
plt.legend()

plt.close("all")


# Now that we have our signal and its timestamps. Let's convert it into a TimeSeries
# Noisy Signal will be the Input and Without Noise Signal the output

# Prepare the input
input_signal = TimeSeries.from_values(noisy_signal)

# Prepare the output
output_signal = TimeSeries.from_values(signal)

## Filtering and predicting using the Gaussian Process filter
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF
from darts import TimeSeries
from darts.models import GaussianProcessFilter

# Infer the mean of a Gaussian Process using a periodic kernel

kernel = ExpSineSquared()
# kernel = RBF()

gpf = GaussianProcessFilter(
    kernel=kernel, alpha=0.09 / 2, n_restarts_optimizer=100
)
filtered_signal = gpf.filter(input_signal)

plt.figure(figsize=[12, 8])
#output_signal.plot(color="black", label="Orginal sine wave")
input_signal.plot(color="red", label="Noisy sine wave")
filtered_signal.plot(color="blue", label="Filtered sine wave")
plt.legend()
plt.show()