from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imghdr
import cv2
import os


data_directory = "../data"
batch_size = 32
image_size = (255, 255)

# Create the image dataset
data = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    batch_size=batch_size,
    image_size=image_size,
    validation_split=0.2,
    seed=42,
    subset="training"
)

print(data.class_names)

train_data = int(len(data) * .6)
val_data = int(len(data) * .2)
test_data = int(len(data) * .2)

train = data.take(train_data)
val = data.skip(train_data).take(val_data)
test = data.skip(train_data + val_data).take(test_data)

for inputs, labesls in train:
    print(inputs.shape)
    print(labesls.shape)
    break

# Feature Engineering

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
inputs, labels = next(iter(normalized_ds))
first_image = inputs[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = data.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 3

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


hist = model.fit(
    train_ds,
    validation_data=val,
    epochs=12
)

# Model Summary

model.summary()

fig = plt.figure()
plt.plot(hist.history["loss"], color="teal", label="loss")
plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
fig.suptitle("loss", fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history["accuracy"], color="teal", label="accuracy")
plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")
fig.suptitle("Accuracy", fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Model Evaluation
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()


for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

print(f'Precision Score:{precision.result().numpy()}, Recall Score:{recall.result().numpy()}, Accuracy Score: {accuracy.result().numpy()}')


# Testing Model
image = cv2.imread("/835466f5e2fefd510af21f0140563f16.jpg")
plt.imshow(image)
plt.show()
