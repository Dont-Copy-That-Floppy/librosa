y, sr = librosa.load(librosa.util.example_audio_file(), duration=5, offset=10)
S = np.abs(librosa.stft(y))
mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)

# Compare the results visually

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max, top_db=None),
                         y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Original STFT')
plt.subplot(2,1,2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_inv - S),
                                                 ref=S.max(), top_db=None),
                         vmax=0, y_axis='log', x_axis='time', cmap='magma')
plt.title('Residual error (dB)')
plt.colorbar()
plt.tight_layout()
plt.show()
