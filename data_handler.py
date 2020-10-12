
import pandas as pd
import numpy as np
import random
import soundfile as sf
from pydub import AudioSegment

# for Speaker
PATH_TRAIN = 'family_train.csv'
PATH_TEST = 'family_test.csv'
BASE_PATH = 'family'
CHUNK_DIRECTORY = 'family-chunks'
CHUNK_TRAIN = 'chunked_train.csv'
CHUNK_TEST = 'chunked_test.csv'
CHUNK_SIZE = 1000  # ms
SAMPLE_RATE = 16000 

# for Sekeh
# PATH_TRAIN = 'sekeh/coin_train.csv'
# PATH_TEST = 'sekeh/coin_test.csv'
# BASE_PATH = 'sekeh/data'
# CHUNK_DIRECTORY = 'sekeh/data-chunks'
# CHUNK_TRAIN = 'sekeh/coin_chunk_train.csv'
# CHUNK_TEST = 'sekeh/coin_chunk_test.csv'
# CHUNK_SIZE = 500  # ms
# SAMPLE_RATE = 16000 


class DataHandler:







    def read_data(self, file_type):
        if file_type == 'original':
            df_train = pd.read_csv(PATH_TRAIN)
            df_test = pd.read_csv(PATH_TEST)
        elif file_type == 'chunk':
            df_train = pd.read_csv(CHUNK_TRAIN)
            df_test = pd.read_csv(CHUNK_TEST)
        else:
            raise Exception('file_type should be \'original\' or \'chunk\'')

        return df_train, df_test

    def create_dataframe(self):
        df = pd.DataFrame(columns=['index, class, path'])

        return df

    def detect_start_and_end_silence_place(self, sound, silence_threshold=-50.0, chunk_size=10):
        trim_ms = 0  # ms
        #chunk_size in ms

        assert chunk_size > 0  # to avoid infinite loop
        while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size

        return trim_ms

    def remove_silence(self, sound):
        start_trim = self.detect_start_and_end_silence_place(sound)
        end_trim = self.detect_start_and_end_silence_place(sound.reverse())
        duration = len(sound)
        trimmed_sound = sound[start_trim:duration - end_trim]

        return trimmed_sound

    def split_sound_file_to_chunks(self, chunk_size=CHUNK_SIZE):
        # chunk_size in ms
        train, test = self.read_data('original')
        self.make_chunks(train, CHUNK_TRAIN, chunk_size)
        self.make_chunks(test, CHUNK_TEST, chunk_size)


    def make_chunks(self, samples, path, chunk_size):
        lst = []
        for sample in samples.values:
            sound = AudioSegment.from_wav(BASE_PATH + sample[2])
            sound = sound.set_frame_rate(SAMPLE_RATE)
            sound = sound.set_channels(1)
            sound = self.remove_silence(sound)  # remove silence from beginning and end of sound
            num_chunks = int(sound.duration_seconds // (chunk_size/1000))
            for i in range(num_chunks):
                t1 = i * chunk_size
                t2 = chunk_size + t1

                chunk_of_sound = sound[t1:t2]
                chunk_of_sound.export(CHUNK_DIRECTORY + sample[2] + '-chunk-' + str(i),
                                      format="wav")  # Exports to a wav file in the current path.
                lst.append([sample[1], sample[2] + '-chunk-' + str(i)])
        chunk_dataframe = pd.DataFrame(lst, columns=['class', 'path'])
        chunk_dataframe.to_csv(path)


    def get_samples(self):
        train, test = self.read_data('chunk')
        train_sounds = self.read_raw_sound(train)
        test_sounds = self.read_raw_sound(test)

        return train_sounds, test_sounds

    def read_raw_sound(self, samples):
        raw_data_with_classes = []
        for sample in samples.values:
            instance, samplerate = sf.read(CHUNK_DIRECTORY+sample[2])
            raw_data_with_classes.append([instance, sample[1]])

        return raw_data_with_classes



























dh = DataHandler()
dh.split_sound_file_to_chunks()
# train, test = dh.get_samples()
s=0













































