""" ImageGenerator with data augmentation Class
Using ImageGenerator Class, we can load files batch by batch.

This module demonstrates
- Different pre-processing method available for training purpose. It creates all
  the data variation copy in RAM and does not store the data in disk. So, it is
  very fast compare to load data from disk.

- Functional API. https://www.tensorflow.org/guide/keras/functional

- How to save and load model.

"""

# %%
from glob import glob
import os.path

from helper.plot import fit_curves
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# %%
DATA_PATH = os.path.join(os.path.dirname(__file__), '../dataset/horse-or-human')
MODEL_SAVE_PATH = 'tmp/cats_and_dogs.h5'
if not os.path.exists('tmp'):
  os.mkdir('tmp')


# %%
def fn_model():
  clear_session()
  # input layer
  i = Input(shape=(300, 300, 3))
  # first convolution layer
  x = Conv2D(16, (3, 3), activation='relu')(i)
  x = MaxPooling2D(2)(x)
  # second convolution layer
  x = Conv2D(32, (3, 3), activation='relu')(x)
  x = MaxPooling2D(2)(x)
  # third convolution layer
  x = Conv2D(64, (3, 3), activation='relu')(x)
  x = MaxPooling2D(2)(x)
  # forth convolution layer
  x = Conv2D(64, (3, 3), activation='relu')(x)
  x = MaxPooling2D(2)(x)
  # fifth convolution layer
  x = Conv2D(128, (3, 3), activation='relu')(x)
  x = MaxPooling2D(2)(x)
  # deep neural network for classification
  x = Flatten()(x)
  x = Dense(512, activation='relu')(x)
  o = Dense(1, activation='sigmoid')(x)

  md = Model(i, o)
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

  train_data_generator = ImageDataGenerator(
      rescale=1.0 / 255,
      rotation_range=50,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
  )  # Apply data augmentation
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
r = model.fit(dataset[0],
              steps_per_epoch=8,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
              epochs=15,
              verbose=1,
              validation_data=dataset[1],
              validation_steps=8)
fit_curves(r)

# model complete save
model.save(MODEL_SAVE_PATH)

# %% load saved model and test on an image
clear_session()
# load model here
md = load_model(MODEL_SAVE_PATH)

image = img_to_array(load_img(glob(f'{DATA_PATH}/train/horses/*')[0])) / 255.0
print(md.predict(np.array([image]))[0])
