""" Transfer Learning
is the reuse of a pre-trained model on a new problem. In this module we took
a portion of InceptionV3 model to create our new model.

**** MUST UNDERSTAND
make sure to mark the existing layers weight to not trainable. Based on dataset,
we can train some bottom layers. If all the layers are trainable, then model
will recreate all the weights, then transfer of learning will not happen.

"""

# %%

from glob import glob
import os

from helper.plot import fit_curves
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_PATH = os.path.join(os.path.dirname(__file__), '../dataset/horse-or-human')
WEIGHT = os.path.join(
    os.path.dirname(__file__),
    '../pre_trained_model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
)


# %% model definition
def fn_model():
  clear_session()
  inception = InceptionV3(input_shape=(150, 150, 3),
                          include_top=False,
                          weights=None)
  inception.load_weights(WEIGHT)
  for layer in inception.layers:
    layer.trainable = False
  inception.summary()

  x = inception.get_layer('mixed7')
  x = Flatten()(x.output)
  x = Dense(512, activation='relu')(x)
  output = Dense(1, activation='sigmoid')(x)
  model = Model(inception.input, output)
  model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(learning_rate=0.0001),
                metrics=['accuracy'])

  model.summary()
  return model


# %% get dataset
def fn_dataset(batch_size=32):
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
      target_size=(150, 150),  # All images will be resized to 300x300
      batch_size=batch_size,
      class_mode='binary',  # To use binary_crossentropy loss, labels = binary
  )

  # validation dataset
  print('validation dataset count =>')
  print('\t horse', len(glob(f'{DATA_PATH}/validation/horses/*')))
  print('\t humans', len(glob(f'{DATA_PATH}/validation/humans/*')))

  validation_data_generator = ImageDataGenerator(rescale=1.0 / 255)
  validation_data = validation_data_generator.flow_from_directory(
      os.path.join(DATA_PATH, 'validation'),  # Directory for validation images
      target_size=(150, 150),  # All images will be resized to 300x300
      batch_size=batch_size,
      class_mode='binary',  # To use binary_crossentropy loss, labels = binary
  )

  return train_data, validation_data


# %% train model
model = fn_model()
dataset = fn_dataset(batch_size=16)
r = model.fit(dataset[0],
              steps_per_epoch=8,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
              epochs=25,
              verbose=1,
              validation_data=dataset[1],
              validation_steps=8)
fit_curves(r)
