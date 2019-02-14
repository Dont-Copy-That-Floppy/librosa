# From time-series input

y, sr = librosa.load(librosa.util.example_audio_file())
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
spec_bw
# array([[ 3379.878,  1429.486, ...,  3235.214,  3080.148]])

# From spectrogram input

S, phase = librosa.magphase(librosa.stft(y=y))
librosa.feature.spectral_bandwidth(S=S)
# array([[ 3379.878,  1429.486, ...,  3235.214,  3080.148]])

# Using variable bin center frequencies

if_gram, D = librosa.ifgram(y)
librosa.feature.spectral_bandwidth(S=np.abs(D), freq=if_gram)
# array([[ 3380.011,  1429.11 , ...,  3235.22 ,  3080.148]])

# Plot the result

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(spec_bw.T, label='Spectral bandwidth')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, spec_bw.shape[-1]])
plt.legend()
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
                         y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()
