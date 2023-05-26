import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data import load_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from visualization import plot_history, plot_confusion_matrix, plot_prediction

# add paths to tensorflow-intel and the root folder of your project
sys.path.append(os.path.join(os.getcwd(), 'env', 'Lib', 'site-packages'))
sys.path.append(os.getcwd())

# Set GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load preprocessed data
X_train, y_train, X_val, y_val, X_test, y_test = load_data()


print(np.unique(y_train))

# Convert to OHE
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the learning rate schedule
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 75:
        lr = 0.0005
    if epoch > 100:
        lr = 0.0003
    return lr

# Define the weight decay parameter
weight_decay = 1e-4

# Apply data augmentation to the training set
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    fill_mode='nearest')

# Reshape input data to have dimensions (num_samples, height, width, channels)
img_height, img_width, num_channels = 32, 32, 3
X_train = X_train.reshape((-1, img_height, img_width, num_channels)).astype('float32')
X_val = X_val.reshape((-1, img_height, img_width, num_channels)).astype('float32')
X_test = X_test.reshape((-1, img_height, img_width, num_channels)).astype('float32')

# Mean shift and variance centering
print(X_train.shape)
mean = np.mean(X_train, axis = (0,1,2,3))
std = np.std(X_train, axis = (0,1,2,3))
X_train = (X_train-mean)/(std + 1e-7)
X_test = (X_test-mean)/(std + 1e-7)
X_val = (X_val-mean)/(std + 1e-7)

train_datagen.fit(X_train)

# Rescale pixel values to [0, 1] range
# X_train = X_train / 255.0
# X_val = X_val / 255.0
# X_test = X_test / 255.0

# Define the model architecture
model = keras.Sequential()
model.build(input_shape=(None, img_height, img_width, num_channels))
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), input_shape=X_train.shape[1:]))
model.add(keras.layers.Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(keras.layers.Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units=10, activation='softmax'))

model.summary()

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping and model checkpointing
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

# Fit the model with data augmentation
history = model.fit(X_train, y_train, batch_size=128, epochs=50, callbacks=[early_stopping, checkpoint], validation_data=(X_val, y_val), shuffle=True)

# Evaluate the model on the test set
model.load_weights('best_model.h5')
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Plot training history
plot_history(history, save_plot=True)
print("saving plot history")
# Make predictions on validation set
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_original = np.argmax(y_val, axis=1)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Plot confusion matrix
plot_confusion_matrix(y_val_original, y_pred_classes, class_names)
print("saving plot confusion matrix")

# Plot sample predictions
plot_prediction(X_val[:5], y_val[:5], y_pred_classes[:5])
print("saving sample predictions")

# Save the model weights
model.save_weights('trained_model.h5')
print("saving the model weights")