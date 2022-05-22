"""Support Functions"""
import matplotlib.pyplot as plt


def fit_curves(r, accuracy=False):
  # print loss curve
  plt.plot(r.history['loss'], label='loss')
  plt.plot(r.history['val_loss'], label='val_loss')
  plt.legend()

  # print accuracy curve
  if accuracy:
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()

  # show the plot
  plt.show()
