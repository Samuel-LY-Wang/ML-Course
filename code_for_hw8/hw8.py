from code_for_hw8_keras import *

import numpy as np

from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D

# classes=2
# layer1 = Dense(units=3, activation='relu', use_bias=False)
# a, b, model = run_keras_2d("3class", archs(3)[0], 10, split=0.5, display=False, verbose=False, trials=1)

train, validation = get_MNIST_data()
train_20, validation_20 = get_MNIST_data(shift=20)
# print(len(train), len(validation))
train = train[0]/255, train[1]
validation = validation[0]/255, validation[1]
train_20 = train_20[0]/255, train_20[1]
validation_20 = validation_20[0]/255, validation_20[1]

layers1 = [Dense(input_dim = 48*48, units=512, activation='relu'),
            Dense(units=256,  activation='relu'),
            Dense(units=10, activation='softmax')
]

layers2 = [Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
          MaxPooling2D(pool_size=(2, 2)),
          Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
          MaxPooling2D(pool_size=(2, 2)),
          Flatten(),
          Dense(units=128, activation='relu'),
          Dropout(0.5),
          Dense(units=10, activation='softmax')
]
epochs = 1
run_keras_fc_mnist(train_20, validation_20, layers1, epochs, split=0.1, trials=5)
run_keras_cnn_mnist(train_20, validation_20, layers2, epochs, split=0.1, trials=5)