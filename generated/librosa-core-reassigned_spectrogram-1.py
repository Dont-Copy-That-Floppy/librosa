y, sr = librosa.load(librosa.util.example_audio_file())
y_zoom = y[27 * sr : 29 * sr]

freqs, times, mags = librosa.reassigned_spectrogram(
    y=y_zoom, sr=sr, hop_length=16, n_fft=64, ref_power=1e-4
)
db = librosa.amplitude_to_db(mags, ref=np.max)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
librosa.display.specshow(
    db, x_axis="s", y_axis="linear", sr=sr, hop_length=16
)
plt.title("Spectrogram")
plt.subplot(2, 1, 2)
plt.scatter(times, freqs, c=db, s=0.1, cmap="magma")
plt.title("Reassigned spectrogram")
plt.xlim([0, 2])
plt.xticks([0, 0.5, 1, 1.5, 2])
plt.ylabel("Hz")
plt.subplots_adjust(
    left=0.1, bottom=0.05, right=0.95, top=0.95, hspace=0.5
)
plt.show()
