# Cough and Sneeze detector

import numpy as np
from pydub import AudioSegment
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Conv1D
from keras.layers import GRU, BatchNormalization
from keras.models import Model, load_model
from utilities import *
from microphone_input import take_input


def create_model(input_shape):
    X_input = Input(shape=input_shape)

    # Step 1: CONV layer : for extracting features
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)  # Batch normalization
    X = Activation('relu')(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)  # Batch normalization

    # Step 3: Second GRU Layer
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer 
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)

    return model


def detect_sickSound(filename, model):
    plt.rcParams['image.cmap'] = 'gray'
    plt.subplot(2, 1, 1)
    x = graph_spectrogram(filename)
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0], color='green')
    plt.ylabel('Probability')
    plt.savefig('output//audio.png')
    return predictions

# Preprocess the audio to the correct format
def preprocess_input(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')


def main():
    Tx = 5511
    Ty = 1375
    n_freq = 101

    model = create_model(input_shape=(Tx, n_freq))
    model.summary()
    model = load_model('./model/tr_model.h5')

    file_name = take_input()

    preprocess_input(file_name)
    prediction = detect_sickSound(file_name, model)

    consecutive_timesteps=0
    threshold= 0.2
    for i in range(Ty):
        consecutive_timesteps += 1
        if prediction[0,i,0] > threshold and consecutive_timesteps > 75:
            print("you just coughed/sneezed")

if __name__ == "__main__":
    main()
