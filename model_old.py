# create a CNN as baseline model for number classification (MNIST Dataset)
import tensorflow as tf
import tensorflow_probability as tfp


def model_baseline(input_shape):
    input_img = tf.keras.Input(shape=input_shape)
    # first layer
    z1 = tf.keras.layers.Conv2D(filters=28, kernel_size=5, strides=1)(input_img)
    a1 = tf.keras.layers.ReLU()(z1)
    p1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(a1)
    # second layer
    z2 = tf.keras.layers.Conv2D(filters=50, kernel_size=5, strides=1)(p1)
    a2 = tf.keras.layers.ReLU()(z2)
    p2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(a2)
    # flatten
    f = tf.keras.layers.Flatten()(p2)
    # third layer
    fc = tf.keras.layers.Dense(units=500, activation='relu')(f)
    # output layer
    outputs = tf.keras.layers.Dense(units=10, activation='softmax')(fc)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


def model_dropout(input_shape, dropt_rate):
    input_img = tf.keras.Input(shape=input_shape)
    # first layer
    z1 = tf.keras.layers.Conv2D(filters=28, kernel_size=5, strides=1)(input_img)
    a1 = tf.keras.layers.ReLU()(z1)
    p1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(a1)
    # second layer
    z2 = tf.keras.layers.Conv2D(filters=50, kernel_size=5, strides=1)(p1)
    a2 = tf.keras.layers.ReLU()(z2)
    p2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(a2)
    # flatten
    f = tf.keras.layers.Flatten()(p2)
    # third layer
    fc = tf.keras.layers.Dense(units=500, activation='relu')(f)
    # add dropout layer here
    drop = tf.keras.layers.Dropout(dropt_rate)(fc)
    # output layer
    outputs = tf.keras.layers.Dense(units=10, activation='softmax')(drop)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


def model_bbb(input_shape):
    input_img = tf.keras.Input(shape=input_shape)
    # first layer
    a1 = tfp.layers.Convolution2DReparameterization(filters=28, kernel_size=5, strides=1, activation='relu')(input_img)
    p1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(a1)
    # second layer
    a2 = tfp.layers.Convolution2DReparameterization(filters=50, kernel_size=5, strides=1, activation='relu')(p1)
    p2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(a2)
    # flatten
    f = tf.keras.layers.Flatten()(p2)
    # third layer
    fc = tf.keras.layers.Dense(units=500, activation='relu')(f)
    # add dropout layer here
    # drop = tf.keras.layers.Dropout(dropt_rate)(fc)
    # output layer
    outputs = tf.keras.layers.Dense(units=10, activation='softmax')(fc)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


model_test = model_bbb((28, 28, 1))
print(model_test.summary())
