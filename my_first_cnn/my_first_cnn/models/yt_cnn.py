from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import imghdr
import cv2
import os


data_dir = "../data"
# inap_image_filter = ["jpeg", "jpg", "png"]

# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         image_path = os.path.join(data_dir, image_class, image)
#         try:
#             img = cv2.imread(image_path)
#             tip = imghdr.what(image_path)
#             if tip not in inap_image_filter:
#                 print("File type is not supported {}".format(image_path))
#                 os.remove(image_path)
#         except Exception as e:
#             print("Issue removing files: {}".format(image_path))

data = tf.keras.utils.image_dataset_from_directory("../data")
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

scaled = batch[0] / 255
print("Min value:",  scaled.min())
print("Max value:",  scaled.max())

data = data.map(lambda x, y: (x / 255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
print(batch[0].max())

print(data)

train_size = int(len(data) * .6)
val_size = int(len(data) * .2)
test_size = int(len(data) * .2)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile("adam", loss=tf.losses.BinaryCrossentropy(),
              metrics=["accuracy"])
model.summary()

log_dir = "./logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

hist = model.fit(train, epochs=20, validation_data=val,
                 callbacks=[tensorboard_callback])
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
