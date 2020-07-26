import librosa
import numpy as np
import pickle
import pandas as pd

SR = 48000


def get_features_from_timeframes(y, label_1, label_2, sr):
    print('y', y.size)

    # with open(path, 'rb') as f:
    #
    #     data = pickle.load(f)

    # voice_1 = [1]*len(y)
    # voice_2 = [0]*len(y)
    # data = np.array([y, voice_1, voice_2])

    # df = pd.DataFrame(data)

    # df = df.rolling(window=1000).mean()
    # print(df.columns, df.shape)
    # exit(0)

    # y = data[0, :]
    # nbr_voices = y.shape[0] - 1

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    S, phase = librosa.magphase(librosa.stft(y))
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)

    nbr_timeframes = mfcc.shape[1]
    window_size = int(len(y) / nbr_timeframes)
    label_1_mean = []
    label_2_mean = []

    # for i in range(0, len(label_1), window_size):
    for i in range(0, len(y), window_size):
        label_1_mean.append(np.mean(label_1[i:i + window_size]))
        label_2_mean.append(np.mean(label_2[i:i + window_size]))

    label_1_mean = label_1_mean[:-1]
    label_2_mean = label_2_mean[:-1]

    print('label_1', len(label_1_mean))
    print('label_2', len(label_2_mean))

    print('MFCC')
    print(mfcc.shape)
    # print(mfcc)

    print('\nZero crossing rate')
    print(zero_crossing_rate.shape)
    # print(zero_crossing_rate)

    print('\nSpectral rolloff')
    print(spectral_rolloff.shape)
    # print(spectral_rolloff)

    features = mfcc

    # print(features.shape)
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

    s1, _ = librosa.load('data/trimmed/victor_5.wav', sr=SR)
    s2, _ = librosa.load('data/trimmed/isak_1.wav', sr=SR)

    full_s, l1, l2 = data_augmentaton(s1, s2, sample_length=1, sr=SR)
    features = get_features_from_timeframes(y=full_s, label_1=l1, label_2=l2, sr=SR)

    plt.plot(full_s)
    plt.plot(l1)
    plt.plot(l2)
    plt.show()

# TODO: Classify
# TODO: Implement more features



