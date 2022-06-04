""" ImageGenerator Class
Using ImageGenerator Class, we can load files batch by batch.
It internally handled by TensorFlow.

This module also introduce 2 callbacks
- EarlyStopping: given the patience value, if validation error does not improve
  it stops the training method. It can also restore the best weight from last
  epochs. Please check the arguments for details

- ModelCheckPoint: after each epoch it creates a model checkpoint in provided
  directory. checkpoint can be complete model or only weights based on argument.
  Also, it is possible to store only the better than last model. Check the docs.

"""

# %%

from glob import glob
import os

from helper.plot import fit_curves
from sklearn import datasets
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_PATH = os.path.join(os.path.dirname(__file__), '../dataset/horse-or-human')
MODEL_SAVE_PATH = 'tmp/checkpoint'
if not os.path.exists('tmp'):
  os.mkdir('tmp')


# %%
def fn_model():
  clear_session()
  md = Sequential([
      # The input shape is 300x300 with 3 bytes color
      Input(shape=(300, 300, 3)),
      # This is the first convolution
      Conv2D(16, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      # The second convolution
      Conv2D(32, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      # The third convolution
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      # The fourth convolution
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      # The fifth convolution
      Conv2D(128, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      # Flatten the results to feed into a DNN
      Flatten(),
      # 512 neuron hidden layer
      Dense(512, activation='relu'),
      # Only 1 output neuron.
      # It will contain a value from 0-1 where 0 for 1 class ('horses') and
      # 1 for the other ('humans')
      Dense(1, activation='sigmoid')
  ])

  md.compile(loss='binary_crossentropy',
             optimizer=RMSprop(learning_rate=0.0001),
             metrics=['accuracy'])
  return md


# %% get dataset
def fn_dataset():
  # training data
  print('train dataset count =>')
  print('\t horse', len(glob(f'{DATA_PATH}/train/horses/*')))
  print('\t humans', len(glob(f'{DATA_PATH}/train/humans/*')))

  train_data_generator = ImageDataGenerator(rescale=1.0 / 255)
  train_data = train_data_generator.flow_from_directory(
      os.path.join(DATA_PATH, 'train'),  # Source directory for training images
      target_size=(300, 300),  # All images will be resized to 300x300
      batch_size=32,
      class_mode='binary',  # To use binary_crossentropy loss, labels = binary
  )

  # validation dataset
  print('validation dataset count =>')
  print('\t horse', len(glob(f'{DATA_PATH}/validation/horses/*')))
  print('\t humans', len(glob(f'{DATA_PATH}/validation/humans/*')))

  validation_data_generator = ImageDataGenerator(rescale=1.0 / 255)
  validation_data = validation_data_generator.flow_from_directory(
      os.path.join(DATA_PATH, 'validation'),  # Directory for validation images
      target_size=(300, 300),  # All images will be resized to 300x300
      batch_size=32,
      class_mode='binary',  # To use binary_crossentropy loss, labels = binary
  )

  return train_data, validation_data


# %% train model
dataset = fn_dataset()
model = fn_model()
model.summary()
r = model.fit(
    dataset[0],
    steps_per_epoch=8,
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint(MODEL_SAVE_PATH,
                        save_weights_only=True,
                        save_best_only=True),
    ],
    epochs=50,
    verbose=1,
    validation_data=dataset[1],
    validation_steps=8,
)
fit_curves(r)
