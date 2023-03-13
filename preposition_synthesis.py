import os
import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# Add relevant paths
AUDIOSET_FILE_PATH = '' # Download from https://research.google.com/audioset/
AUDIOCAPS_METADATA_PATH = '' # Download from https://github.com/cdjkim/audiocaps/blob/master/dataset/train.csv
OUTPUT_PATH = ''
PREPOSITIONS = set(['before', 'after', 'then', 'followed by'])

SAMPLING_RATE = 16000
NUM = 50000


def generate_sentence(sentence_1, sentence_2):
    preposition = random.sample(PREPOSITIONS, 1)[0]
    if preposition == 'after':
        return '{} {} {}'.format(sentence_2.lower(), preposition, sentence_1.lower()).capitalize().replace('/', ' ')
    else:
        return '{} {} {}'.format(sentence_1.lower(), preposition, sentence_2.lower()).capitalize().replace('/', ' ')


def concat_audio(file_1, file_2, crossfade=1, max_len=20): # AudioSet maximum length 10s
    audio_1, _ = librosa.load(file_1, sr=SAMPLING_RATE)
    audio_2, _ = librosa.load(file_2, sr=SAMPLING_RATE)

    fade_out = np.concatenate((np.ones(audio_1.size-SAMPLING_RATE*crossfade), np.linspace(1, 0, num=SAMPLING_RATE*crossfade)))
    fade_in = np.concatenate((np.linspace(0, 1, num=SAMPLING_RATE*crossfade), np.ones(audio_2.size-SAMPLING_RATE*crossfade)))

    faded_audio_1 = audio_1 * fade_out
    faded_audio_2 = audio_2 * fade_in
    return np.pad(faded_audio_1, (0, SAMPLING_RATE*max_len - faded_audio_1.size)) + np.pad(faded_audio_2, (faded_audio_1.size - SAMPLING_RATE*crossfade, SAMPLING_RATE*max_len - (faded_audio_1.size + faded_audio_2.size - SAMPLING_RATE*crossfade)))


if __name__ == '__main__':
    metadata_df = pd.read_csv(AUDIOCAPS_METADATA_PATH)

    metadata = []
    for m in metadata_df.values:
        audioset_id = '{}_{}'.format(m[1], m[2]) # Assuming AudioSet file name as '{youtube_id}_{start_time}.wav'
        metadata.append(('{}/{}.wav'.format(AUDIOSET_FILE_PATH, audioset_id), m[3]))

    no_preposition = [m for m in metadata if not any(p in m[1].lower() for p in PREPOSITIONS)]
    pairs = [random.sample(no_preposition, 2) for _ in range(NUM)]

    for (file_1, sentence_1), (file_2, sentence_2) in pairs:
        sentence = generate_sentence(sentence_1, sentence_2)        
        audio = concat_audio(file_1, file_2)

        of = '{}/{}.wav'.format(OUTPUT_PATH, sentence)
        if not os.path.exists(of):
            sf.write(of, audio, SAMPLING_RATE, 'PCM_16')