"""multi label classification
Multi-label classification involves predicting zero or more class labels

Unlike normal classification tasks where class labels are mutually exclusive, multi-label classification requires specialized machine learning algorithms that support predicting multiple mutually non-exclusive classes or "labels".
"""

# %%

from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# %% define dataset
x, y = make_multilabel_classification(n_samples=1000,
                                      n_features=20,
                                      n_classes=3,
                                      n_labels=2,
                                      random_state=12)
# summarize dataset shape
print(x.shape, y.shape)
# summarize first few examples
for i in range(10):
  print(x[i], y[i])


# %% model definition
def fn_model(n_inputs, n_outputs):
  clear_session()
  md = Sequential([
      Dense(32,
            input_dim=n_inputs,
            kernel_initializer='he_uniform',
            activation='relu'),
      Dense(n_outputs, activation='sigmoid'),
  ])

  md.compile(loss='binary_crossentropy', optimizer='adam')
  return md


# %% train model
model = fn_model(20, 3)
model.summary()
model.fit(x, y, epochs=400)

# %% evaluate model
y_hat = model.predict(x)
y_hat = y_hat.round()
# calculate accuracy
print(f'model accuracy {accuracy_score(y, y_hat)}')
