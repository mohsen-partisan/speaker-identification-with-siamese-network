
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
from data_handler import DataHandler
from data_handler import SAMPLE_RATE
from data_handler import CHUNK_SIZE


DATA_TEST_PATH = 'data_test.csv'
RAW_SOUND_PATH = 'test_data'
REFINED_SOUND_PATH = 'test_data/data_test_refined'
QUERY_DATA_PATH = 'query_data.csv'
QUERY_SOUND_PATH = 'test_data/query_sample'
REFINED_QUERY_SOUND_PATH = 'test_data/refined_query_sample'


class TestModel:

    def read_data_test(self, path):
        data = pd.read_csv(path)

        return data

    # call this method just once
    def read_and_preprocess_raw_sound(self, data, raw_sound_path, refined_sound_path):
        for sample in data.values:
            sound = AudioSegment.from_wav(raw_sound_path + sample[2])
            sound = DataHandler.remove_silence(DataHandler(), sound)
            sound.export(refined_sound_path + sample[2],
                                  format="wav")

    def get_samples(self, data, path):
        raw_data = []
        for sample in data.values:
            instance, samplerate = sf.read(path + sample[2])
            raw_data.append(instance)
            print(len(instance))

        return raw_data

    def add_padding_to_samples(self, raw_data):
        # arr = np.array(raw_data)
        padded_samples = pad_sequences(raw_data, dtype='float32', maxlen = SAMPLE_RATE*(CHUNK_SIZE//1000))
        return padded_samples

    def create_pairs_test(self,query_sample ,samples):
        pairs = []
        for i in range(len(samples)):
            pairs += [[query_sample[0], samples[i]]]

        return np.array(pairs)



    def load_model(self):
        model = keras.models.load_model('saved_model/speaker.h5', compile=False)

        return model

    # this method extract calculated difference between two siamese networks
    # for rank users based on this differentiate
    def access_last_layer(self, data_test, model):
        last_layer = model.layers[-1].output
        keras_function = keras.backend.function([model.input], [last_layer])
        last_layer_output = keras_function([[data_test[:, 0], data_test[:, 1]], 1])
        last_layer_output = np.array(last_layer_output)

        return last_layer_output

    def find_best_class(self, distances):
        closest_id = np.argmin(distances)+1

        return closest_id







test = TestModel()
data = test.read_data_test(DATA_TEST_PATH)
test.read_and_preprocess_raw_sound(data, RAW_SOUND_PATH, REFINED_SOUND_PATH)
samples = test.get_samples(data, REFINED_SOUND_PATH)
padded_samples = test.add_padding_to_samples(samples)
query_data = test.read_data_test(QUERY_DATA_PATH)
test.read_and_preprocess_raw_sound(query_data, QUERY_SOUND_PATH, REFINED_QUERY_SOUND_PATH)
query_sample = test.get_samples(query_data, REFINED_QUERY_SOUND_PATH)
padded_query_sample = test.add_padding_to_samples(query_sample)
data_test = test.create_pairs_test(padded_query_sample, padded_samples)
model = test.load_model()
distances = test.access_last_layer(data_test, model)
closest_class = test.find_best_class(distances)
print(distances)
print('prediction class is: ')
print(closest_class)