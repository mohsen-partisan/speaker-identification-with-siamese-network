'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for more details).
# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras import layers
from keras import Sequential
from keras.optimizers import RMSprop
from keras import backend as K
from data_handler_all import DataHandler
from data_handler_all import CHUNK_SIZE

num_classes = 8
epochs = 100
SAMPLING_RATE = 48000
FILTERS = 320
EMBEDDING_DIMENSION = 128
BATCH_SIZE = 12


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def calculate_input_shape():
    chunk_size_in_second = CHUNK_SIZE / 1000 # ms to second
    return int(chunk_size_in_second * SAMPLING_RATE)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    y_true = float(y_true)
    y_pred = float(y_pred)
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape, filters, embedding_dimension, dropout=0.05):
    '''Base network to be shared (eq. to feature extraction).
    '''
    encoder = Sequential()

    encoder.add(layers.Conv1D(filters, 32, padding='same', activation='relu', input_shape=input_shape))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D(4, 4))

    # Further convs
    encoder.add(layers.Conv1D(2 * filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.Conv1D(3 * filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.Conv1D(4 * filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.GlobalMaxPool1D())

    encoder.add(layers.Dense(embedding_dimension))

    return encoder




    # input = Input(shape=input_shape)
    # x = Flatten()(input)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # x = Dense(128, activation='relu')(x)
    # return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


samples = DataHandler().get_samples()
random.shuffle(samples)
y_train = []
x_train = []
for i in range(len(samples)):
    y_train.append(samples[i][1])
    x_train.append(samples[i][0])
x_train = np.array(x_train)
y_train = np.array(y_train)

input_shape = [calculate_input_shape(), 1]

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == (i+1))[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)


# network definition
base_network = create_base_network(input_shape, FILTERS, EMBEDDING_DIMENSION)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
# output = layers.Dense(1, activation='sigmoid')(distance)
model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=BATCH_SIZE,
          epochs=epochs,
          )
model.save('/content/gdrive/My Drive/speaker-identification/new-siamese network/siamese network/saved_model/my_model.h5')

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
