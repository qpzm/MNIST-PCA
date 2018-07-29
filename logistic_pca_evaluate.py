from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed
from keras.utils import multi_gpu_model
import pickle
import os
from os import path


input_dim = 784  # 28*28
batch_size = 128 * 4
n_classes = 10
summary_file = './summary'
seed(1)
set_random_seed(2)


def build_logistic_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
    return model

def show_train_history(history):
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def concat_params(kernel, bias):
    bias = bias.reshape((1,10))
    return np.concatenate((kernel, bias), axis=0).flatten()


def train(dir, optimizer='SGD', norm='maxmin', lr=1e-2, data='fashion_mnist', epochs=100):
    # the data, shuffled and split between train and test sets
    if data == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, input_dim)
    X_test = X_test.reshape(10000, input_dim)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    if norm == 'maxmin':
        X_train /= 255
        X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)

    model = build_logistic_model(input_dim, n_classes)
    kernel, bias = model.get_layer(index=0).get_weights()
    init_weight = concat_params(kernel, bias)

    model.summary()

    model = multi_gpu_model(model)
    filepath = "{dir}/{{epoch:01d}}.hdf5".format(dir=dir)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=False, mode='max')
    if not os.path.exists(dir):
        os.mkdir(dir)
    callbacks_list = [checkpoint]

    model.compile(optimizer=getattr(optimizers,optimizer)(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    init_loss = model.evaluate(X_train, Y_train)
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=0, validation_data=(X_test, Y_test), callbacks=callbacks_list)

    train_acc = history.history['acc']
    test_acc = history.history['val_acc']
    with open(summary_file, 'a') as f:
        f.write("%s\n%s %s\n" % (dir, train_acc[-1], test_acc[-1]))

    return init_weight, init_loss, history


def pca(dir, h5_group_path, weights, epochs):
    for i in range(1, epochs + 1):
        with h5py.File(f'{dir}/{i}.hdf5', 'r') as f:
            kernel = f[ h5_group_path + 'kernel:0'].value
            bias =  f[ h5_group_path + 'bias:0'].value
            weights.append(concat_params(kernel, bias))

    weights = np.stack(weights, axis=0)
    u, s, vh = np.linalg.svd(weights, full_matrices=False)
    coordinates = np.dot(weights, vh[0:10].T)
    return s[0:10], coordinates


def calc(data, optimizer, norm, lr, epochs, iteration):
    dir = f'{data}-weights-{optimizer}-{lr}-{norm}'
    init_weight, init_loss, history = train(dir=dir, data=data,
                                            optimizer=optimizer,
                                            norm=norm, lr=lr, epochs=epochs)

    coordinates = []
    weights = [np.array(init_weight)]
    s, coordinates = pca(dir, f'model_weights/sequential_{iteration}/dense_{iteration}/',
                      weights, epochs)

    with open(f'{dir}/store.pkl', 'wb') as f:
        pickle.dump([coordinates, history.history, init_loss], f)
    with open(f'{dir}/sv.pkl', 'wb') as f:
        pickle.dump(s, f)
