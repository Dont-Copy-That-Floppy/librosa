# Compute local onset autocorrelation
y, sr = librosa.load(librosa.util.example_audio_file())
hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                      hop_length=hop_length)
# Compute global onset autocorrelation
ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
ac_global = librosa.util.normalize(ac_global)
# Estimate the global tempo for display purposes
tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=hop_length)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.subplot(3, 1, 1)
plt.plot(oenv, label='Onset strength')
plt.xticks([])
plt.legend(frameon=True)
plt.axis('tight')
plt.subplot(3, 1, 2)
# We'll truncate the display to a narrower range of tempi
librosa.display.specshow(tempogram[:100], sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='tempo',
                         tmin=tempo/4, tmax=2*tempo, n_yticks=4)
plt.subplot(3, 1, 3)
x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr, num=tempogram.shape[0])
plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
plt.plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
plt.xlabel('Lag (seconds)')
plt.axis('tight')
plt.legend(frameon=True)
plt.tight_layout()
