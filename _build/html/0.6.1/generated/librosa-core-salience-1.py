y, sr = librosa.load(librosa.util.example_audio_file(),
                     duration=15, offset=30)
S = np.abs(librosa.stft(y))
freqs = librosa.core.fft_frequencies(sr)
harms = [1, 2, 3, 4]
weights = [1.0, 0.5, 0.33, 0.25]
S_sal = librosa.salience(S, freqs, harms, weights, fill_value=0)
print(S_sal.shape)
# (1025, 646)
import matplotlib.pyplot as plt
plt.figure()
librosa.display.specshow(librosa.amplitude_to_db(S_sal,
                                                 ref=np.max),
                         sr=sr, y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Salience spectrogram')
plt.tight_layout()
