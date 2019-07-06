# From time-series input

y, sr = librosa.load(librosa.util.example_audio_file())
# Approximate maximum frequencies with roll_percent=0.85 (default)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
rolloff
# array([[ 8376.416,   968.994, ...,  8925.513,  9108.545]])
# Approximate minimum frequencies with roll_percent=0.1
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.1)
rolloff
# array([[ 75.36621094,  64.59960938,  64.59960938, ...,  75.36621094,
# 75.36621094,  64.59960938]])

# From spectrogram input

S, phase = librosa.magphase(librosa.stft(y))
librosa.feature.spectral_rolloff(S=S, sr=sr)
# array([[ 8376.416,   968.994, ...,  8925.513,  9108.545]])

# With a higher roll percentage:
y, sr = librosa.load(librosa.util.example_audio_file())
librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
# array([[ 10012.939,   3003.882, ...,  10034.473,  10077.539]])

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(rolloff.T, label='Roll-off frequency')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, rolloff.shape[-1]])
plt.legend()
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()
plt.show()
