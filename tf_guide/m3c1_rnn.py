""" RNN
Simple RNN Code
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# %%
series = np.sin(0.1 * np.arange(200)) + np.random.randn(200) * 0.1
plt.plot(series)
plt.show()

# %%
# build the dataset
# let's see if we can use T past values to predict the next value
T = 10
X = []
Y = []
for t in range(len(series) - T):
  x = series[t:t + T]
  X.append(x)
  y = series[t + T]
  Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print('X.shape', X.shape, 'Y.shape', Y.shape)

# %%
backend.clear_session()
md_v1 = Sequential([layers.Input(shape=(T,)), layers.Dense(1)])
md_v1.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
r = md_v1.fit(
    X[:-N // 2],
    Y[:-N // 2],
    epochs=200,
    validation_data=(X[-N // 2:], Y[-N // 2:]),
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# %%
# "Wrong" forecast using true targets
validation_target = Y[-N // 2:]
validation_predictions = []

# index of first validation input
i = -N // 2

while len(validation_predictions) < len(validation_target):
  p = md_v1.predict(X[i].reshape(1, -1))[0, 0]  # 1x1 array -> scalar
  i += 1
  # update the predictions list
  validation_predictions.append(p)

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()

# %%
# Forecast future values
# use only self-predictions for making future predictions
validation_target = Y[-N // 2:]
validation_predictions = []

# first validation input
last_x = X[-N // 2]  # 1-D array of length T

while len(validation_predictions) < len(validation_target):
  p = md_v1.predict(last_x.reshape(1, -1))[0, 0]  # 1x1 array -> scalar
  # update the predictions list
  validation_predictions.append(p)
  # make the new input
  last_x = np.roll(last_x, -1)
  last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()
