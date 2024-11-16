import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import tensorflow as tf
import random

# Fixed random seed
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Image size and training parameters
img_width, img_height = 224, 224
batch_size = 64
epochs = 50

# Data augmentation
regularized_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)

# Speed to class mapping
speed_to_class = {0.0025: 0, 0.005: 1, 0.01: 2, 0.02: 3, 0.04: 4}

# Build classification model
def build_classification_model(regularized=False):
    reg = l2(0.005) if regularized else None
    dropout_rate = 0.5 if regularized else 0.0

    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=(img_width, img_height, 3), kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Conv2D(64, (5, 5), padding="same", kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Conv2D(128, (5, 5), padding="same", kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(5, activation="softmax"))

    return model

# Validate file paths
def check_file_path(file_path, file_type="path"):
    if not os.path.exists(file_path):
        print(f"Error: {file_type.capitalize()} '{file_path}' does not exist.")
        return False
    return True

# Load training images with labels
def load_images_with_classes(paths_and_labels, img_size=(img_width, img_height)):
    images, labels = [], []
    for path, label in paths_and_labels.items():
        if not check_file_path(path, "directory"):
            continue
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            try:
                image = load_img(img_path, target_size=img_size)
                image = img_to_array(image) / 255.0
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Skipping file: {filename}, error: {e}")
    return np.array(images), np.array(labels)

# Load test images from Excel with labels
def load_test_images_from_excel(test_path, labels_file, img_size=(img_width, img_height)):
    labels_df = pd.read_excel(labels_file)
    images, labels = [], []

    for index, row in labels_df.iterrows():
        filename = row['filename']
        speed = row['speed']

        img_path = os.path.join(test_path, filename)
        if not check_file_path(img_path, "file"):
            continue

        try:
            image = load_img(img_path, target_size=img_size)
            image = img_to_array(image) / 255.0
            images.append(image)

            # Map speed to class label
            labels.append(speed_to_class[speed])
        except Exception as e:
            print(f"Skipping file: {filename}, error: {e}")

    return np.array(images), np.array(labels)

# Train classification model
def train_classification_model(model, datagen, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer=Adam(learning_rate=0.00001),
                  loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])

    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        steps_per_epoch=len(train_images) // batch_size,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        verbose=1
    )

    return history

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# Paths to training data
train_data = {
    'D:/CNN/data/train/velocity_0.0025': 0.0025,
    'D:/CNN/data/train/velocity_0.005': 0.005,
    'D:/CNN/data/train/velocity_0.02': 0.02,
    'D:/CNN/data/train/velocity_0.04': 0.04,
    'D:/CNN/data/train/velocity_0.01': 0.01
}
train_images, train_labels = load_images_with_classes(train_data)
train_labels = to_categorical([speed_to_class[label] for label in train_labels])

# Paths to test data
test_path = 'D:/CNN/data/test'
test_labels_file = 'D:/CNN/data/test/test_images_speeds.xlsx'
test_images, test_labels = load_test_images_from_excel(test_path, test_labels_file)
test_labels = to_categorical(test_labels)

# Build classification model
classification_model = build_classification_model(regularized=True)

# Train model
classification_history = train_classification_model(
    classification_model,
    regularized_datagen,
    train_images,
    train_labels,
    test_images,
    test_labels
)

# Plot training history
plot_training_history(classification_history)

# Predict on test set
test_predictions = np.argmax(classification_model.predict(test_images), axis=1)

# Classification report
print("Classification Report:")
print(classification_report(np.argmax(test_labels, axis=1), test_predictions, target_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]))
