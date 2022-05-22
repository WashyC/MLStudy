""" Binary Classification using TensorFlow
No hidden layer
"""

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from helper.plot import fit_curves
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential

# %%
# load the data
data = load_breast_cancer()
print(data.keys())
print(data.data.shape, data.target_names, data.feature_names)

x_train, x_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    test_size=0.3)
N, D = x_train.shape
print(N, D)

# %%
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# %%
md_classification = Sequential([
    Input((D,)),
    Dense(1, activation='sigmoid'),
])

md_classification.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
r = md_classification.fit(x_train,
                          y_train,
                          validation_data=(x_test, y_test),
                          epochs=100)
fit_curves(r, accuracy=True)  # plot loss
# %%
# Evaluate the model - evaluate() returns loss and accuracy
print('Train score:', md_classification.evaluate(x_train, y_train))
print('Test score:', md_classification.evaluate(x_test, y_test))

# %%
# predict
pred = md_classification.predict(x_test)
print(pred)
pred = np.round(pred).flatten()
print(pred)

# %%
# Calculate the accuracy, compare it to evaluate() output
print('Manually calculated accuracy:', np.mean(pred == y_test))
print('Evaluate output:', md_classification.evaluate(x_test, y_test))
