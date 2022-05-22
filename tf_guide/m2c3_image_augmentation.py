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
def fn_get_model():
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
  # deep nural network for classification
  x = Flatten()(x)
  x = Dense(512, activation='relu')(x)
  o = Dense(1, activation='sigmoid')(x)

  md = Model(i, o)
  md.compile(loss='binary_crossentropy',
             optimizer=RMSprop(learning_rate=0.0001),
             metrics=['accuracy'])
  return md


# %%
def fn_train():
  clear_session()
  # training data
  print('train dataset count =>')
  print('\t cat:', len(glob(f'{DATA_PATH}/train/cats/*')))
  print('\t dog:', len(glob(f'{DATA_PATH}/train/dogs/*')))

  # validation dataset
  print('validation dataset count =>')
  print('\t cat:', len(glob(f'{DATA_PATH}/validation/cats/*')))
  print('\t dog:', len(glob(f'{DATA_PATH}/validation/dogs/*')))

  train_datagen = ImageDataGenerator(
      rescale=1.0 / 255,
      rotation_range=50,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
  )  # Apply data augmentation
  train_generator = train_datagen.flow_from_directory(f'{DATA_PATH}/train',
                                                      target_size=(300, 300),
                                                      batch_size=32,
                                                      class_mode='binary')

  valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
  valid_generator = valid_datagen.flow_from_directory(f'{DATA_PATH}/validation',
                                                      target_size=(300, 300),
                                                      batch_size=32,
                                                      class_mode='binary')

  model = fn_get_model()
  model.summary()
  r = model.fit(
      train_generator,
      steps_per_epoch=8,
      callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
      epochs=15,
      verbose=1,
      validation_data=valid_generator,
      validation_steps=8)
  fit_curves(r)

  # model complete save
  model.save(MODEL_SAVE_PATH)


def fn_test():
  clear_session()
  image = img_to_array(load_img(glob(f'{DATA_PATH}/train/cats/*')[0]))
  # load model here
  model = load_model(MODEL_SAVE_PATH)
  print(model.predict([image])[0])


if __name__ == '__main__':
  fn_train()
