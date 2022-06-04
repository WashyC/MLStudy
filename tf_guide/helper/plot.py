"""Support Functions"""
import matplotlib.pyplot as plt


def fit_curves(history, accuracy=False):
  # print loss curve
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.legend()

  # print accuracy curve
  if accuracy:
    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()

  # show the plot
  plt.show()
