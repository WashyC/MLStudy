""" multiclass_classification.py
without any hidden layer
"""

# %%
from helper.plot import fit_curves
from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential

# %% load and normalize the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('x_train.shape:', x_train.shape)


# %% model definition
def fn_model():
  clear_session()
  md = Sequential([
      Flatten(input_shape=(28, 28)),
      Dense(128, activation='relu'),
      Dense(10, activation='softmax'),
  ])
  md.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
  return md


# %% configure and train
md_classification = fn_model()
md_classification.summary()
history = md_classification.fit(x_train,
                                y_train,
                                validation_data=(x_test, y_test),
                                epochs=10)
fit_curves(history, accuracy=True)

# %% evaluate test data
print(md_classification.evaluate(x_test, y_test))
