import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from keras.callbacks import LearningRateScheduler

def plot_history(history, save_plot=False):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # plot accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_plot:
        plt.savefig('accuracy_plot.png')
    else:
        plt.show()
    
    plt.figure()
    # plot loss
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_plot:
        print("Current working directory:", os.getcwd())  # Print current working directory
        plt.savefig('loss_plot.png')
        print("Loss plot saved successfully.")
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_plot=False):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.grid(False)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    if save_plot:
        plt.savefig('confusion_matrix.png')
    else:
        plt.show()


def plot_prediction(x, y_true, y_pred, save_plot=False):
    fig, axs = plt.subplots(1, len(x), figsize=(10, 10))
    for i in range(len(x)):
        axs[i].imshow(x[i])
        axs[i].axis('off')
        axs[i].set_title(f'True: {y_true[i]}, Predicted: {y_pred[i]:.2f}')
    if save_plot:
        plt.savefig('prediction_plot.png')
    else:
        plt.show()
