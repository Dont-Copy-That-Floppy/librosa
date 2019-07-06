# A basic STFT inverse example

y, sr = librosa.load(librosa.util.example_audio_file(), duration=5, offset=30)
# Get the magnitude spectrogram
S = np.abs(librosa.stft(y))
# Invert using Griffin-Lim
y_inv = librosa.griffinlim(S)
# Invert without estimating phase
y_istft = librosa.istft(S)

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
librosa.display.waveplot(y_istft, sr=sr, color='r')
plt.title('Magnitude-only istft reconstruction')
plt.tight_layout()
plt.show()
