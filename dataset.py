
def convert_audio_files_to_mfcc(file_path):
    import librosa

    y, sr = librosa.load(file_path, offset=0)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    S, phase = librosa.magphase(librosa.stft(y))
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)

    print('MFCC')
    print(mfcc)

    print('\nZero crossing rate')
    print(zero_crossing_rate)

    print('\nSpectral rolloff')
    print(spectral_rolloff)

    return mfcc


