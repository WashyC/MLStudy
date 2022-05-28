""" cnn.py
"""
# %%
from matplotlib import pyplot as plt
from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential

from helper.plot import fit_curves

# %% prepare the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('x_train.shape:', x_train.shape)
print('y_train.shape:', y_train.shape)
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[1])
plt.show()

# %% model definition
optimizer, loss = 'adam', 'sparse_categorical_crossentropy'
metric = ['accuracy']


# simple network without any hidden layer
def fn_simple_model():
  clear_session()
  md = Sequential([
      Flatten(input_shape=(28, 28)),
      Dense(10, activation='softmax'),
  ])
  md.summary()
  md.compile(optimizer=optimizer, loss=loss, metrics=metric)
  r = md.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)
  fit_curves(r)
  md.evaluate(x_test, y_test)


# deep nural network with one hidden network
def fn_ann_model():
  clear_session()
  md = Sequential([
      Flatten(input_shape=(28, 28)),
      Dense(512, activation='relu'),
      Dense(10, activation='softmax'),
  ])
  md.summary()
  md.compile(optimizer=optimizer, loss=loss, metrics=metric)
  r = md.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)
  fit_curves(r)
  md.evaluate(x_test, y_test)


# convolutional nural network
# cnn for feature extraction
# one hidden network for classification
def fn_cnn_model():
  clear_session()
  md = Sequential([
      Reshape((28, 28, 1), input_shape=(28, 28)),
      Conv2D(32, (3, 3), (2, 2), activation='relu'),
      Conv2D(64, (3, 3), 2, activation='relu'),
      Conv2D(128, (3, 3), 2, activation='relu'),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(10, activation='softmax')
  ])
  md.summary()
  md.compile(optimizer=optimizer, loss=loss, metrics=metric)
  r = md.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)
  fit_curves(r)
  md.evaluate(x_test, y_test)


# %%
if __name__ == '__main__':
  # %%
  fn_simple_model()
  # %%
  fn_ann_model()
  # %%
  fn_cnn_model()
