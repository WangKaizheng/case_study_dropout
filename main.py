import numpy as np
from keras.datasets import mnist
from model import model_baseline
from keras.utils import np_utils


def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    y_test = np_utils.to_categorical(y_test, 10)
    y_train = np_utils.to_categorical(y_train, 10)

    return (x_train, y_train), (x_test, y_test)


# load the data
(train_images, train_labels), (test_images, test_labels) = prepare_data()

# prepare the model
model = model_baseline(train_images.shape[1:])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# train the model
model.fit(x=train_images,
          y=train_labels,
          batch_size=32,
          epochs=2,
          validation_data=(test_images, test_labels))

# evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print('loss: {}'.format(loss))
print('accuracy: {}'.format(accuracy))
