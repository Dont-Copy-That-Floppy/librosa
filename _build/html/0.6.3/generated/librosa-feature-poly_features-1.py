y, sr = librosa.load(librosa.util.example_audio_file())
S = np.abs(librosa.stft(y))

# Fit a degree-0 polynomial (constant) to each frame

p0 = librosa.feature.poly_features(S=S, order=0)

# Fit a linear polynomial to each frame

p1 = librosa.feature.poly_features(S=S, order=1)

# Fit a quadratic to each frame

p2 = librosa.feature.poly_features(S=S, order=2)

# Plot the results for comparison

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
ax = plt.subplot(4,1,1)
plt.plot(p2[2], label='order=2', alpha=0.8)
plt.plot(p1[1], label='order=1', alpha=0.8)
plt.plot(p0[0], label='order=0', alpha=0.8)
plt.xticks([])
plt.ylabel('Constant')
plt.legend()
plt.subplot(4,1,2, sharex=ax)
plt.plot(p2[1], label='order=2', alpha=0.8)
plt.plot(p1[0], label='order=1', alpha=0.8)
plt.xticks([])
plt.ylabel('Linear')
plt.subplot(4,1,3, sharex=ax)
plt.plot(p2[0], label='order=2', alpha=0.8)
plt.xticks([])
plt.ylabel('Quadratic')
plt.subplot(4,1,4, sharex=ax)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log')
plt.tight_layout()
