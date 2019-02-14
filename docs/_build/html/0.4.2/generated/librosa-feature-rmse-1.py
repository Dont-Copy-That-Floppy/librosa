y, sr = librosa.load(librosa.util.example_audio_file())
librosa.feature.rmse(y=y)
# array([[ 0.   ,  0.056, ...,  0.   ,  0.   ]], dtype=float32)

# Or from spectrogram input

S, phase = librosa.magphase(librosa.stft(y))
rms = librosa.feature.rmse(S=S)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(rms.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, rms.shape[-1]])
plt.legend(loc='best')
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
                         y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()
