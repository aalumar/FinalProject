from typing import Dict, List
import matplotlib.pyplot as plt

def plot_loss_curves(results: Dict[str, List[float]]):
  """
  Plots a training curve of a results dictionary
  """

  # get loss values of dictionary
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  # get accuracy values of dictionary
  accuracy = results["train_accuracy"]
  test_accuracy = results["test_accuracy"]

  # how many epochs
  epochs = range(len(results["train_loss"]))

  # set up plot
  plt.figure(figsize=(15, 7))

  # plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  # plot accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label="train_accuracy")
  plt.plot(epochs, test_accuracy, label="test_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()
