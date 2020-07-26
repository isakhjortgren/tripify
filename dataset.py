
def convert_audio_files_to_mfcc(path):
    import librosa
    import numpy as np
    import pickle
    import pandas as pd

    # TODO: Use the timeseries aspect with an LSTM instead of mean values?
    y, sr = librosa.load(path)
    print('y', y.size)
    sr = 48000

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
    # mfcc = np.mean(mfcc.T, axis=0)

    nbr_timeframes = mfcc.shape[1]

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    # zero_crossing_rate = np.mean(zero_crossing_rate)

    S, phase = librosa.magphase(librosa.stft(y))
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    # spectral_rolloff = np.mean(spectral_rolloff)


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


