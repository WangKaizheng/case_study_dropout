from model_comparison import sgd_dropout
from prepare_data import prepare_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# gets rid of an error about cpu on Macbook Pro 2017
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_model(model_name, dropout_type, dropout_rate, train_parameter):
    # prepare images & labels from dataset
    (x_train, y_train), (x_test, y_test) = prepare_data()
    if model_name == 'sgd_dropout':
        model = sgd_dropout(x_train.shape[1:], dropout_type, dropout_rate)
        # configure model before training
        tf.keras.optimizers.SGD(learning_rate=train_parameter[0], name='SGD')
        model.compile(optimizer='SGD', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # train the model for a fixed number of epochs
        history = model.fit(
            x=x_train, y=y_train, validation_data=(x_test, y_test),
            epochs=train_parameter[1], batch_size=train_parameter[2])
        # calculate training & validation accuracy
        train_acc_store = history.history['accuracy']
        train_err = 100.0 - 100.0 * (train_acc_store[-1])
        val_acc_store = history.history['val_accuracy']
        val_acc = 100.0 - 100.0 * (val_acc_store[-1])
        error_set = [train_err, val_acc, dropout_rate]
        acc_store = [train_acc_store, val_acc_store]
        # calculate training & validation loss
        train_loss = history.history['loss'][-1]
        train_loss_store = history.history['loss']
        val_loss = history.history['val_loss'][-1]
        val_loss_store = history.history['val_loss']
        loss_set = [train_loss, val_loss]
        loss_store = [train_loss_store, val_loss_store]
        show_training_figure(acc_store, loss_store, dropout_type, dropout_rate)
    else:
        print('INVALID TRAINING')
        history = False
        error_set = False
        loss_set = False
    return history, error_set, loss_set


def show_training_figure(acc_store, loss_store, p_distribution, p):
    figure_name = p_distribution + ' p = ' + str(p)
    plt.figure(num=figure_name)
    plt.subplot(2, 1, 1)
    plt.plot(acc_store[0], linestyle='-', color='b')
    plt.plot(acc_store[1], linestyle='--', color='r')
    plt.title('Accuracy under ' + p_distribution + ' p = ' + str(p))
    # plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(loss_store[0], linestyle='-', color='b')
    plt.plot(loss_store[1], linestyle='--', color='r')
    plt.title('Loss under ' + p_distribution + ' p = ' + str(p))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='lower right')
    # plt.show()


def evaluate_different_p():
    # set hyperperameter for the model
    hparameter = (0.01, 10, 32)
    # lists to store ploting values locally
    # for Bernoulli:
    test_error_bern = []
    train_error_bern = []
    train_loss_bern = []
    test_loss_bern = []
    # for Gaussian:
    test_error_gaus = []
    train_error_gaus = []
    test_loss_gaus = []
    train_loss_gaus = []
    # returns the list of drop_rate(p) from 0.1-0.9, into list
    drop_rate = list(np.linspace(0.1, 0.9, 9))
    print(drop_rate)
    # run dropout on all values of p
    for i in range(0, len(drop_rate)):
        # Store error values for Bernoulli
        history, error_bernoulli, loss_bernoulli = \
            train_model('sgd_dropout', 'Bernoulli', drop_rate[i], hparameter)
        test_error_bern.append(error_bernoulli[0])
        train_error_bern.append(error_bernoulli[1])
        test_loss_bern.append(loss_bernoulli[0])
        train_loss_bern.append(loss_bernoulli[1])
        # show_training_figure(history, 'Bernoulli', drop_rate[i])
        print(train_error_bern)
        print("\n")
    # Store error values for Gaussian
    for i in range(0, len(drop_rate)):
        history, error_gaussian, loss_gaussian = \
            train_model('sgd_dropout', 'Gaussian', drop_rate[i], hparameter)
        # Store error values for figure b
        test_error_gaus.append(error_gaussian[0])
        train_error_gaus.append(error_gaussian[1])
        test_loss_gaus.append(loss_gaussian[0])
        train_loss_gaus.append(loss_gaussian[1])
        # show_training_figure(history, 'Gaussian', drop_rate[i])
    # Figure A: effect of p under Bernoulli distribution
    plt.figure(num='effect of p')
    plt.subplot(2, 2, 1)
    plt.ylabel('Classification Error %')
    plt.xlabel('Probability of retaining a unit (p) under Bernoulli')
    plt.plot(drop_rate, test_error_bern, label="Test Error")
    plt.plot(drop_rate, train_error_bern, label="Training Error")
    plt.legend(loc='upper right')
    # Figure B: effect of p under Gaussian distribution
    plt.subplot(2, 2, 2)
    plt.ylabel('Classification Error %')
    plt.xlabel('Probability of retaining a unit (p) under Gaussian')
    plt.plot(drop_rate, test_error_gaus, label="Test Error")
    plt.plot(drop_rate, train_error_gaus, label="Training Error")
    plt.legend(loc='upper right')
    # Figure C: effect of p under Bernoulli distribution
    plt.subplot(2, 2, 3)
    plt.ylabel('Loss')
    plt.xlabel('Probability of retaining a unit (p) under Bernoulli')
    plt.plot(drop_rate, test_loss_bern, label="Test Loss")
    plt.plot(drop_rate, train_loss_bern, label="Training Loss")
    plt.legend(loc='upper right')
    # Figure D: effect of p under Gaussian distribution
    plt.subplot(2, 2, 4)
    plt.ylabel('Loss')
    plt.xlabel('Probability of retaining a unit (p) under Gaussian')
    plt.plot(drop_rate, test_loss_gaus, label="Test Loss")
    plt.plot(drop_rate, train_loss_gaus, label="Training Loss")
    plt.legend(loc='upper right')
    plt.tight_layout()
    # save plotted dropout figure
    plt.show()


evaluate_different_p()
