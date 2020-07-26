
def convert_audio_files_to_mfcc(path):
    import librosa
    import numpy as np
    import pickle


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


