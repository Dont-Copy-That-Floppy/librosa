# Compute tonnetz features from the harmonic component of a song

y, sr = librosa.load(librosa.util.example_audio_file())
y = librosa.effects.harmonic(y)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
tonnetz
# array([[-0.073, -0.053, ..., -0.054, -0.073],
# [ 0.001,  0.001, ..., -0.054, -0.062],
# ...,
# [ 0.039,  0.034, ...,  0.044,  0.064],
# [ 0.005,  0.002, ...,  0.011,  0.017]])

# Compare the tonnetz features to `chroma_cqt`

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
librosa.display.specshow(tonnetz, y_axis='tonnetz')
plt.colorbar()
plt.title('Tonal Centroids (Tonnetz)')
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.feature.chroma_cqt(y, sr=sr),
                         y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chroma')
plt.tight_layout()
