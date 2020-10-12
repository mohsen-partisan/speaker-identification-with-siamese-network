
import pandas as pd
import numpy as np
import random
import soundfile as sf
from pydub import AudioSegment

PATH = '/home/mohsen/speaker identification/siamese network/family_siamese.csv'
BASE_PATH = '/home/mohsen/Desktop/family'
CHUNK_DIRECTORY = 'family-chunks'
CHUNK_PATH = 'chunked_family.csv'
CHUNK_SIZE = 1000  # ms


class DataHandler:







    def read_data(self, file_type):
        if file_type == 'original':
            df = pd.read_csv(PATH)
        elif file_type == 'chunk':
            df = pd.read_csv(CHUNK_PATH)
        else:
            raise Exception('file_type should be \'original\' or \'chunk\'')

        return df

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
        samples = self.read_data('original')
        lst = []
        for sample in samples.values:
            sound = AudioSegment.from_wav(BASE_PATH + sample[2])
            sound = self.remove_silence(sound)  # remove silence from beginning and end of sound
            for i in range(0,int(sound.duration_seconds)):
                t1 = i * chunk_size
                t2 = chunk_size+t1

                chunk_of_sound = sound[t1:t2]
                chunk_of_sound.export(CHUNK_DIRECTORY + sample[2] + '-chunk-'+str(i), format="wav")  # Exports to a wav file in the current path.
                lst.append([sample[1], sample[2] + '-chunk-'+str(i)])
        chunk_dataframe = pd.DataFrame(lst, columns=['class', 'path'])
        chunk_dataframe.to_csv(CHUNK_PATH)



    def get_samples(self):
        raw_data_with_classes = []
        samples = self.read_data('chunk')
        for sample in samples.values:
            instance, samplerate = sf.read(CHUNK_DIRECTORY+sample[2])
            raw_data_with_classes.append([instance, sample[1]])


        return raw_data_with_classes


























dh = DataHandler()
# dh.split_sound_file_to_chunks()
data = dh.get_samples()
s=0













































