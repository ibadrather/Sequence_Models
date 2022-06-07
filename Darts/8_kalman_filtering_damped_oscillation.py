# Author: Ibad Rather, ibad.rather.ir@gmail.com, June 2022

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from utils import clear_terminal
from utils import *

from darts import TimeSeries
from darts.models import KalmanFilter

clear_terminal()

def add_white_noise(signal):
    sigma = 0.009
    return np.random.normal(signal, sigma, signal.shape[0])

# Generate timestamps
signal_length = 20
signal_frequency = 100 # hz
timestamps = np.linspace(0, signal_length, signal_length*signal_frequency)

# Generate Sin wave
omega = 20
signal = np.exp(-timestamps)*np.sin(omega*timestamps)

# Adding Noise to it
noisy_signal = add_white_noise(signal)
plt.plot(timestamps, signal, color="r", label="Signal")
plt.plot(timestamps, noisy_signal, color="b", label="Noisy Signal")
plt.legend()
plt.show()

plt.close("all")

# Now that we have our signal and its timestamps. Let's convert it into a TimeSeries

# Noisy Signal will be the Input and Without Noise Signal the output

# Prepare the input
input_signal = TimeSeries.from_values(noisy_signal)

# Prepare the output
output_signal = TimeSeries.from_values(signal)

# Let's visualise these TimeSeries Signals
plt.figure(figsize=[12, 8])
input_signal.plot(label="Noisy Input")
output_signal.plot(color="gray", label="Ideal Output")
plt.legend()

# Traning
kf = KalmanFilter(dim_x=1)
kf.fit(output_signal, input_signal)

# Filtering
signal_filtered = kf.filter(output_signal, input_signal)
plt.close("all")
plt.figure(figsize=[12, 8])
input_signal.plot(label="Noisy Input")
output_signal.plot(color="gray", label="Ideal Output")
signal_filtered.plot(color="red", label="Filtered Output")
plt.legend()
plt.show()

