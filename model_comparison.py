# Kaizheng Wang
import tensorflow as tf

# Model Referrence:
# sgd_droupt() ===>> Dropout: A Simple Way to Prevent Neural Networks from Overfitting


def sgd_dropout(input_shape, dropout_type, dropout_rate):
    input_img = tf.keras.Input(shape=input_shape)
    # first layer
    x = tf.keras.layers.Conv2D(28, kernel_size=(3, 3), activation="relu")(input_img)
    # flatten
    x = tf.keras.layers.Flatten()(x)
    # second layer
    # to simplify the calculation: units 256 rather 2048
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    # set dropout rate as 1 => no dropout
    if dropout_type == 'Bernoulli':
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    if dropout_type == 'Gaussian':
        x = tf.keras.layers.GaussianDropout(dropout_rate)(x)
    # third layer
    # to simplify the calculation: units 256 rather 2048
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    if dropout_type == 'Bernoulli':
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    if dropout_type == 'Gaussian':
        x = tf.keras.layers.GaussianDropout(dropout_rate)(x)
    # fourth layer
    # to simplify the calculation: units 256 rather 2048
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    if dropout_type == 'Bernoulli':
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    if dropout_type == 'Gaussian':
        x = tf.keras.layers.GaussianDropout(dropout_rate)(x)
    # output layer
    outputs = tf.keras.layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


# model_test = sgd_dropout((28, 28, 1), 'Bernoulli', 0.5)
# print(model_test.summary())
