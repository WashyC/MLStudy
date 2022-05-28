""" callback.py
"""
# %%
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential

from helper.plot import fit_curves

# %% prepare the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# %% model definition
optimizer, loss = 'adam', 'sparse_categorical_crossentropy'
metric = ['accuracy']


def fn_model():
  clear_session()
  md = Sequential([
      Flatten(input_shape=(28, 28)),
      Dense(1024, activation='relu'),
      Dense(10, activation='softmax'),
  ])
  md.summary()
  md.compile(optimizer=optimizer, loss=loss, metrics=metric)
  return md


# %% callback
class MyCallback(Callback):

  def on_epoch_end(self, epoch, logs=None):
    if logs.get('accuracy') > 0.9:
      print('\nAccuracy is greater than 0.9')
      self.model.stop_training = True


# %%
if __name__ == '__main__':
  # %%
  model = fn_model()
  r = model.fit(x_train,
                y_train,
                validation_data=(x_test, y_test),
                epochs=15,
                callbacks=[MyCallback()])
  fit_curves(r)
  model.evaluate(x_test, y_test)
