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

# 固定随机种子
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 图像尺寸和批次大小设置
img_width, img_height = 224, 224
batch_size = 64
epochs = 25

# 数据增强
regularized_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)

# 定义速度值到分类标签的映射
speed_to_class = {0.0025: 0, 0.005: 1, 0.01: 2, 0.02: 3, 0.04: 4}

# 模型构建函数
def build_classification_model(regularized=False):
    reg = 0.005 if regularized else 0.0
    dropout_rate = 0.5 if regularized else 0.0

    model = Sequential()
    inputShape = (img_width, img_height, 3)

    # 第一卷积层
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout_rate > 0: model.add(Dropout(dropout_rate))

    # 第二卷积层
    model.add(Conv2D(64, (5, 5), padding="same", kernel_regularizer=l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout_rate > 0: model.add(Dropout(dropout_rate))

    # 第三卷积层
    model.add(Conv2D(128, (5, 5), padding="same", kernel_regularizer=l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout_rate > 0: model.add(Dropout(dropout_rate))

    # 全连接层
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    if dropout_rate > 0: model.add(Dropout(dropout_rate))

    # 分类输出层
    model.add(Dense(5, activation="softmax"))  # 5 表示分类类别数
    return model

# 检查文件路径
def check_file_path(file_path, file_type="path"):
    if not os.path.exists(file_path):
        print(f"Error: {file_type.capitalize()} '{file_path}' does not exist.")
        return False
    return True

# 加载训练数据并添加分类标签
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

# 从 Excel 文件加载测试数据并分配分类标签
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

            # 将速度值映射到分类标签
            labels.append(speed_to_class[speed])
        except Exception as e:
            print(f"Skipping file: {filename}, error: {e}")

    return np.array(images), np.array(labels)

# 训练模型
def train_classification_model(model, datagen, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        steps_per_epoch=len(train_images) // batch_size,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        verbose=1
    )

    return history

# 绘制训练过程图
def plot_training_history(history):
    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制准确率图
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# 加载训练数据
train_data = {
    'D:/CNN/data/train/velocity_0.0025': 0.0025,
    'D:/CNN/data/train/velocity_0.005': 0.005,
    'D:/CNN/data/train/velocity_0.02': 0.02,
    'D:/CNN/data/train/velocity_0.04': 0.04,
    'D:/CNN/data/train/velocity_0.01': 0.01
}
train_images, train_labels = load_images_with_classes(train_data)
train_labels = np.array([speed_to_class[label] for label in train_labels])

# 加载测试数据
test_path = 'D:/CNN/data/test'
test_labels_file = 'D:/CNN/data/test/test_images_speeds.xlsx'
test_images, test_labels = load_test_images_from_excel(test_path, test_labels_file)

# 构建分类模型
classification_model = build_classification_model(regularized=True)

# 训练分类模型
classification_history = train_classification_model(
    classification_model,
    regularized_datagen,
    train_images,
    train_labels,
    test_images,
    test_labels
)

# 绘制训练过程图
plot_training_history(classification_history)

# 测试集预测
test_predictions = np.argmax(classification_model.predict(test_images), axis=1)

# 打印分类报告
print("Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]))
