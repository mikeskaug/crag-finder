import os
import pickle

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import regularizers
from keras.callbacks import TensorBoard

import numpy as np
from scipy.misc import imresize

from dataset import get_data_sets, layer_means
from config import LOG_DIR

def cnn_model():
    model = Sequential()

    model.add(Conv2D(input_shape=(128, 128, 7), filters=32, kernel_size=3, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='VarianceScaling', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='VarianceScaling'))
    
    return model


def fbeta(y_true, y_pred, beta=2, threshold_shift=0):
    # ensure that predictions are in the range [0, 1]
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))


def compile_callbacks(logs_subdir=''):
    return [
        TensorBoard(
            log_dir=os.path.join(LOG_DIR, logs_subdir),
            histogram_freq=0,
            write_graph=False,
            write_images=True
        )
    ]


def train(model, batch_size=25, num_epochs=10):
    data = get_data_sets(augment=True)
    (means, variances) = data['train'].layer_mean_variance()

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    def image_mod(batch):
        images, labels = batch
        images = images - means
        images = images / variances
        # downsample 2x
        images = images[:, ::2, ::2, :]
        return (images, labels)

    history = model.fit_generator(generator=data['train'].batch_generator(batch_size, transform=image_mod, loop=True),
                                  steps_per_epoch=data['train'].num_examples / batch_size,
                                  epochs=num_epochs,
                                  validation_data=data['validation'].batch_generator(batch_size, transform=image_mod, loop=True),
                                  validation_steps=data['validation'].num_examples / batch_size,
                                  workers=4,
                                  use_multiprocessing=True,
                                  callbacks=compile_callbacks(logs_subdir='run10-24xaugment'))
    return (model, history)


if __name__ == "__main__":
    model = cnn_model()
    (model, history) = train(model, num_epochs=50)
    model.save('./models/crag-finder-model-run10.h5')
    with open('./models/run10.pickle', 'wb') as out:
        pickle.dump(history.history, out)
