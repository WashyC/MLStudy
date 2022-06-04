""" callback.py
use custom callback to stop training
"""
# %%
from helper.plot import fit_curves
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential

# %% prepare the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# %% model definition
def fn_model():
  clear_session()
  md = Sequential([
      Flatten(input_shape=(28, 28)),
      Dense(1024, activation='relu'),
      Dense(10, activation='softmax'),
  ])
  md.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
  return md


# %% callback
class MyCallback(Callback):

  def on_epoch_end(self, epoch, logs=None):
    if logs.get('accuracy') > 0.9:
      print('\nAccuracy is greater than 0.9')
      self.model.stop_training = True


# %% train model
model = fn_model()
model.summary()
hs = model.fit(x_train,
               y_train,
               validation_data=(x_test, y_test),
               epochs=15,
               callbacks=[MyCallback()])
fit_curves(hs)
model.evaluate(x_test, y_test)
