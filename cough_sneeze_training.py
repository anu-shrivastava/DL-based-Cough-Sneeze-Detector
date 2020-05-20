import numpy as np
from utilities import *
import os
from pydub import AudioSegment
import pandas as pd
import ast
from collections import defaultdict, namedtuple
from model import create_model
from keras.optimizers import Adam
import random
pd.set_option("display.max_columns", None)


def insert_ones(y, segment_end_ms):
    """
    for creating the label vector y. The labels of the 50 output steps strictly after the end of the segment
    are set to 1.

    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 5000.0)

    # last timestep for inserting 1
    end_index = y.shape[1] - 1
    # check if end_index goes out of output timesteps or not
    if segment_end_y + 51 < y.shape[1]:
        end_index = segment_end_y + 51
    # make the required steps as 1
    y[0, segment_end_y + 1:end_index] = 1

    return y

def load_audio(filePath):
    """ loads raw audio from disk

    Arguments:
    filePath -- absolute path of file

    Returns:
    audioClip -- """
    audioClip = AudioSegment.from_wav(filePath)
    return audioClip

def load_csv(csvPath):
    csvData =[]
    if os.path.exists(csvPath):
        csvData = pd.read_csv(csvPath, header = [1])
    else:
        print("file does not exist: ", csvPath)
        exit(-1)
    return csvData

def preprocessAudioFile(wavFile):
    wavFile = match_target_amplitude(wavFile, -20)
    msDuration = len(wavFile)
    #make sure all audio is exactly 5s long. clip/append 0 otherwise
    if msDuration < 5000:
        silence = AudioSegment.silent(duration=(5000-msDuration))
        wavFile += silence
    else:
        wavFile = wavFile[:5000]
    #write modified file to disk as I couldnt find a way to pass an audio segment directly to wavfile
    file_handle = wavFile.export("trainFile" + ".wav", format="wav")
    wavSpec = graph_spectrogram("trainFile.wav")

    return wavSpec

def extractDataFromDF(inputDataFrame):
    pathLblSecs_dict = defaultdict(list)
    anno = namedtuple('anno', 'label end_ms')

    for _, eachRow in inputDataFrame.iterrows():
        filePath = eachRow[0][9:-2]
        endmSec = round(float(eachRow[1])*1000, 2)
        label = ast.literal_eval(eachRow[2])

        pathLblSecs_dict[filePath].append(anno(label, endmSec))

    return pathLblSecs_dict

def get_sickDataAndLabels(annotationCsvPath, outputLabelShape):
    annotationDataDF = load_csv(annotationCsvPath)
    slicedAnnoDataDF = annotationDataDF[['file_list', 'temporal_segment_end', 'metadata']]
    path_LblSecsTup_dict = extractDataFromDF(slicedAnnoDataDF)

    # read and append audio and label for each file in csv
    xList, yList = [], []
    for filePath, annoData in path_LblSecsTup_dict.items():
        x = preprocessAudioFile(load_audio(filePath)).transpose()

        y = np.zeros(outputLabelShape)
        # both cough and sneeze are currently labeled as one category
        for eachEntry in annoData:
            y = insert_ones(y, eachEntry.end_ms)
            # print(filePath, eachEntry.end_ms, eachEntry.label["sick"], y.shape)

        xList.append(x)
        yList.append(y.transpose())

    return xList, yList

def get_not_sickDataAndLabels(audioDir, outputLabelShape):
    xList = []
    yList = []
    for root, _, filenames in os.walk(audioDir):
        if filenames:
            for filename in filenames:
                filepath = os.path.join(root, filename)
                x = preprocessAudioFile(load_audio(filepath)).transpose()
                y = np.zeros(outputLabelShape)
                # print(filepath, 0, y.shape)

                xList.append(x)
                yList.append(y.transpose())

    return xList,yList

def shuffle_data_and_labels(dataList, labelsList):
    tempList = list(zip(dataList, labelsList))
    random.shuffle(tempList)
    dataList, labelsList = zip(*tempList)

    return dataList, labelsList

def get_trainDataAndLabels(annotationCsvPath, negativeSoundsDir, outputLabelShape):
    xList, yList = get_sickDataAndLabels(annotationCsvPath, outputLabelShape)
    tempXList, tempYList = get_not_sickDataAndLabels(negativeSoundsDir, outputLabelShape)
    xList.extend(tempXList)
    yList.extend(tempYList)

    xList, yList = shuffle_data_and_labels(xList, yList)

    X_arr = np.array(xList)
    Y_arr = np.array(yList)

    return X_arr, Y_arr


if __name__ == "__main__":
    annoCSVFilePath_train = r"sampleInput\train\sick\cough_sneeze_detection11May2020_16h42m23s_export.csv"
    negativeSoundsDir_train = r"sampleInput\train\not_sick"
    annoCSVFilePath_dev = r"sampleInput\train\sick\cough_sneeze_detection11May2020_16h42m23s_export.csv"
    negativeSoundsDir_dev = r"sampleInput\train\not_sick"

    Ty = 685
    Tx = 2754
    n_freq = 101

    X, Y = get_trainDataAndLabels(annoCSVFilePath_train, negativeSoundsDir_train, (1, Ty))
    model = create_model(input_shape = (Tx, n_freq))
    print(model.summary())

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.fit(X, Y, batch_size=5, epochs=5)

    # Xdev, Ydev = get_dataAndLabels(annoCSVFilePath_dev, negativeSoundsDir_dev, (1, Ty))
    Xdev, Ydev = X, Y
    loss, acc = model.evaluate(Xdev, Ydev)
    print("hola::accuracy: ", acc)

    model_json = model.to_json()
    with open("./model/newModel.json", 'w') as jsonFile:
        jsonFile.write(model_json)
    model.save_weights("./model/newModel.h5")