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

from helper.plot import fit_curves
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_PATH = 'dataset/cats_and_dogs_filtered'
WEIGHT = 'dataset/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def fn_get_model():
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
  )  # Apply data generation
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


# %%
if __name__ == '__main__':
  fn_train()
