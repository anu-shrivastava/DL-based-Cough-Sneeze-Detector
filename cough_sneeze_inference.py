# Cough and Sneeze detector

import numpy as np
from pydub import AudioSegment, playback
from keras.models import load_model, model_from_json
from utilities import *
from microphone_input import take_input


def detect_sickSound(filename, model):
    plt.rcParams['image.cmap'] = 'gist_earth_r'
    plt.subplot(2, 1, 1)
    x = graph_spectrogram(filename)
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    plt.ylabel('Frequency')

    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0], color='green')
    plt.ylabel('Prediction confidence')
    plt.savefig('output//audio.png')
    return predictions, plt

# Preprocess the audio to the correct format
def preprocess_input(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=5000)
    segment = AudioSegment.from_wav(filename)[:5000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')

def beepOnDetection(threshold, prediction, file_name):
    """
    :param threshold: float value between 0 and 1
    :param prediction: list of detection confidence values for each frame
    :param file_name: path of audio file whoose prediction values are passed
    :return: nothing
    """
    beep_file = "beep.wav"

    Ty = len(prediction[0])
    consecutive_timesteps = 0
    segment = AudioSegment.from_wav(file_name)
    beep = AudioSegment.from_wav(beep_file)
    for i in range(Ty):
        consecutive_timesteps += 1
        if prediction[0, i, 0] > threshold and consecutive_timesteps > 75:
            segment = segment.overlay(beep, position=((i / Ty) * segment.duration_seconds) * 1000)
            print("I am {:.2f}% confident you just coughed/sneezed ".format(prediction[0, i, 0] * 100))
            consecutive_timesteps = 0
    playback.play(segment)
    segment.export("output//beep_output.wav", format='wav')



def main():
    with open("./model/newModel.json", 'r') as json_file:
        model_in_json = json_file.read()
    model = model_from_json(model_in_json)
    model.load_weights('./model/newModel.h5')
    # model.summary()

    file_name = take_input()
    preprocess_input(file_name)
    prediction, plot = detect_sickSound(file_name, model)

    threshold= 0.48
    plot.show()
    beepOnDetection(threshold, prediction, file_name)


if __name__ == "__main__":
    main()
