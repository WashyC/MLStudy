""" binary_classification.py
without any hidden layer
"""

# %%
from helper.plot import fit_curves
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential

# %% load the data
data = load_breast_cancer()
print(data.keys())
print(data.data.shape, data.target_names, data.feature_names)

x_train, x_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    test_size=0.3)
N, D = x_train.shape
print(N, D)

# %% normalize dataset
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# %% model definition
def fn_model():
  clear_session()
  md = Sequential([
      Input((D,)),
      Dense(1, activation='sigmoid'),
  ])
  md.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return md


# %% train the model
md_classification = fn_model()
md_classification.summary()
history = md_classification.fit(x_train,
                                y_train,
                                validation_data=(x_test, y_test),
                                epochs=100)
fit_curves(history, accuracy=True)  # plot loss

# %% model evaluation on train data as well as test data
# evaluate() returns loss and accuracy
print('Train score:', md_classification.evaluate(x_train, y_train))
print('Test score:', md_classification.evaluate(x_test, y_test))

# %% model predict
pred = md_classification.predict(x_test)
print(pred)
pred = np.round(pred).flatten()
print(pred)

# %% calculate the accuracy, compare it to evaluate() output
print('Manually calculated accuracy:', np.mean(pred == y_test))
print('Evaluate output:', md_classification.evaluate(x_test, y_test))
