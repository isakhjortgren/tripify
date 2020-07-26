import librosa
import numpy as np
import pickle

def convert_audio_files_to_mfcc(path):
    # TODO: Use the timeseries aspect with an LSTM instead of mean values?
    # y, sr = librosa.load(path, offset=0)
    sr = 48000

    with open(path, 'rb') as f:
        data = pickle.load(f)

    y = data[0, :]
    nbr_voices = y.shape[0] - 1

    for frame in Y_TIMEFAMES:
        continue
        # TODO: Create variable for different timeframes

        for voice in range(1, nbr_voices+1):
            continue
            # TODO: Round up to 1 for every timeframe


    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # mfcc = np.mean(mfcc.T, axis=0)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    # zero_crossing_rate = np.mean(zero_crossing_rate)

    S, phase = librosa.magphase(librosa.stft(y))
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    # spectral_rolloff = np.mean(spectral_rolloff)

    print(mfcc.shape)
    print(spectral_rolloff.shape)

    print('MFCC')
    print(mfcc)

    print('\nZero crossing rate')
    print(zero_crossing_rate)

    print('\nSpectral rolloff')
    print(spectral_rolloff)

    features = mfcc

    print(features.shape)
    return features


def _insert_sound_random(sound: np.ndarray, label: np.ndarray, full_sequence: np.ndarray):
    size_of_sound = sound.size
    max_len = full_sequence.size - size_of_sound
    random_start = np.random.randint(0, max_len)
    label[random_start:random_start+size_of_sound] = 1
    full_sequence[random_start:random_start + size_of_sound] += sound
    return full_sequence, label


def data_augmentaton(sound_1: np.ndarray, sound_2: np.ndarray, sample_length: int, sr: int):
    full_sequence = np.zeros(sr*sample_length)
    label_1 = np.zeros(full_sequence.size)
    label_2 = np.zeros(full_sequence.size)

    # insert sound_1
    full_sequence, label_1 = _insert_sound_random(sound_1, label_1, full_sequence)

    # insert sound_2
    full_sequence, label_2 = _insert_sound_random(sound_2, label_2, full_sequence)
    return full_sequence, label_1, label_2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = np.linspace(0, 10, 1000)
    s1 = np.sin(t)
    s2 = np.sin(2*t)

    full_s, l1, l2 = data_augmentaton(s1, s2, 30, 100)
    plt.plot(full_s)
    plt.plot(l1)
    plt.plot(l2)
    plt.show()



