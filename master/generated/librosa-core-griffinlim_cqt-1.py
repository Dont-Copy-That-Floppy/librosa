# A basis CQT inverse example

y, sr = librosa.load(librosa.util.example_audio_file(), duration=5, offset=30, sr=None)
# Get the CQT magnitude, 7 octaves at 36 bins per octave
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=36, n_bins=7*36))
# Invert using Griffin-Lim
y_inv = librosa.griffinlim_cqt(C, sr=sr, bins_per_octave=36)
# And invert without estimating phase
y_icqt = librosa.icqt(C, sr=sr, bins_per_octave=36)

# Wave-plot the results

import matplotlib.pyplot as plt
plt.figure()
ax = plt.subplot(3,1,1)
librosa.display.waveplot(y, sr=sr, color='b')
plt.title('Original')
plt.xlabel('')
plt.subplot(3,1,2, sharex=ax, sharey=ax)
librosa.display.waveplot(y_inv, sr=sr, color='g')
plt.title('Griffin-Lim reconstruction')
plt.xlabel('')
plt.subplot(3,1,3, sharex=ax, sharey=ax)
librosa.display.waveplot(y_icqt, sr=sr, color='r')
plt.title('Magnitude-only icqt reconstruction')
plt.tight_layout()
plt.show()
