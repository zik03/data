import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
import tensorflow as tf
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Image dimensions and batch size settings
img_width, img_height = 224, 224
batch_size = 64
epochs = 25

# Data augmentation
regularized_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=15,  # Random rotation
    width_shift_range=0.15,  # Horizontal shift
    height_shift_range=0.15,  # Vertical shift
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True  # Horizontal flip
)

# Map speed values to classification labels
speed_to_class = {0.0025: 0, 0.005: 1, 0.01: 2, 0.02: 3, 0.04: 4}

# Function to build the classification model
def build_classification_model(regularized=False):
    reg = 0.005 if regularized else 0.0  # Regularization parameter
    dropout_rate = 0.5 if regularized else 0.0  # Dropout rate

    model = Sequential()
    inputShape = (img_width, img_height, 3)

    # First convolutional layer
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout_rate > 0: model.add(Dropout(dropout_rate))

    # Second convolutional layer
    model.add(Conv2D(64, (5, 5), padding="same", kernel_regularizer=l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout_rate > 0: model.add(Dropout(dropout_rate))

    # Third convolutional layer
    model.add(Conv2D(128, (5, 5), padding="same", kernel_regularizer=l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout_rate > 0: model.add(Dropout(dropout_rate))

    # Fully connected layer
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    if dropout_rate > 0: model.add(Dropout(dropout_rate))

    # Classification output layer
    model.add(Dense(5, activation="softmax"))  # 5 categories
    return model

# Check if the file path exists
def check_file_path(file_path, file_type="path"):
    if not os.path.exists(file_path):
        print(f"Error: {file_type.capitalize()} '{file_path}' does not exist.")
        return False
    return True

# Load training data and assign classification labels
def load_images_with_classes(paths_and_labels, img_size=(img_width, img_height)):
    images, labels = [], []
    for path, label in paths_and_labels.items():
        if not check_file_path(path, "directory"):
            continue
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            try:
                image = load_img(img_path, target_size=img_size)  # Load image and resize
                image = img_to_array(image) / 255.0  # Normalize pixel values
                images.append(image)
                labels.append(label)  # Assign label
            except Exception as e:
                print(f"Skipping file: {filename}, error: {e}")
    return np.array(images), np.array(labels)

# Load test data from an Excel file and assign classification labels
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
            image = load_img(img_path, target_size=img_size)  # Load image and resize
            image = img_to_array(image) / 255.0  # Normalize pixel values
            images.append(image)

            # Map speed values to classification labels
            labels.append(speed_to_class[speed])
        except Exception as e:
            print(f"Skipping file: {filename}, error: {e}")

    return np.array(images), np.array(labels)

# Train the classification model
def train_classification_model(model, datagen, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",  # Use sparse categorical cross-entropy for integer labels
                  metrics=["accuracy"])  # Track accuracy

    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),  # Training data generator
        steps_per_epoch=len(train_images) // batch_size,  # Steps per epoch
        epochs=epochs,  # Number of epochs
        validation_data=(test_images, test_labels),  # Validation data
        verbose=1
    )

    return history

# Plot the training history
def plot_training_history(history):
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# Load training data
train_data = {
    'D:/CNN/data/train/velocity_0.0025': 0.0025,
    'D:/CNN/data/train/velocity_0.005': 0.005,
    'D:/CNN/data/train/velocity_0.02': 0.02,
    'D:/CNN/data/train/velocity_0.04': 0.04,
    'D:/CNN/data/train/velocity_0.01': 0.01
}
train_images, train_labels = load_images_with_classes(train_data)
train_labels = np.array([speed_to_class[label] for label in train_labels])

# Load test data
test_path = 'D:/CNN/data/test'
test_labels_file = 'D:/CNN/data/test/test_images_speeds.xlsx'
test_images, test_labels = load_test_images_from_excel(test_path, test_labels_file)

# Build the classification model
classification_model = build_classification_model(regularized=True)

# Train the classification model
classification_history = train_classification_model(
    classification_model,
    regularized_datagen,
    train_images,
    train_labels,
    test_images,
    test_labels
)

# Plot the training history
plot_training_history(classification_history)

# Predict on test set
test_predictions = np.argmax(classification_model.predict(test_images), axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]))
