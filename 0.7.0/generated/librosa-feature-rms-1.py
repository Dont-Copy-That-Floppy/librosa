y, sr = librosa.load(librosa.util.example_audio_file())
librosa.feature.rms(y=y)
# array([[ 0.   ,  0.056, ...,  0.   ,  0.   ]], dtype=float32)

# Or from spectrogram input

S, phase = librosa.magphase(librosa.stft(y))
rms = librosa.feature.rms(S=S)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogy(rms.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, rms.shape[-1]])
plt.legend()
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()

# Use a STFT window of constant ones and no frame centering to get consistent
# results with the RMS computed from the audio samples `y`

S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
librosa.feature.rms(S=S)
plt.show()
