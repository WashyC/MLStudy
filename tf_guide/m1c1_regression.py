""" Simple Regression model using TensorFlow
A house has a base cost of 50k, and every additional bedroom adds a cost of 50k.
This will make a 1-bedroom house cost 100k, a 2-bedroom house cost 150k.
y = mx + c, where m = price of each bedroom 50k, c = base price of 50K
"""

# %%
import numpy as np

from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential

# %%
# Dataset
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])

# regression model
# a model can be written in 2 ways
# Sequential and Functional API
# the following patter represent sequential way
md_regression = Sequential([
    Input(shape=(1,)),
    Dense(1),
    # can merge input layer with first dense layer
    # Dense(1, input_shape=(1,))
])

md_regression.summary()

# %%
# compile and train
clear_session()
md_regression.compile(optimizer='sgd', loss='mse')
md_regression.fit(xs, ys, epochs=1000)

# %%
# Prediction
x_new = 7.0
prediction = md_regression.predict([x_new])[0]
print(prediction[0])

print('weights', md_regression.layers[0].weights)
