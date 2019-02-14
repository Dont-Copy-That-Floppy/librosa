# Find nearest neighbors in MFCC space

y, sr = librosa.load(librosa.util.example_audio_file())
mfcc = librosa.feature.mfcc(y=y, sr=sr)
R = librosa.segment.recurrence_matrix(mfcc)

# Or fix the number of nearest neighbors to 5

R = librosa.segment.recurrence_matrix(mfcc, k=5)

# Suppress neighbors within +- 7 samples

R = librosa.segment.recurrence_matrix(mfcc, width=7)

# Use cosine similarity instead of Euclidean distance

R = librosa.segment.recurrence_matrix(mfcc, metric='cosine')

# Require mutual nearest neighbors

R = librosa.segment.recurrence_matrix(mfcc, sym=True)

# Use an affinity matrix instead of binary connectivity

R_aff = librosa.segment.recurrence_matrix(mfcc, mode='affinity')

# Plot the feature and recurrence matrices

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(R, x_axis='time', y_axis='time')
plt.title('Binary recurrence (symmetric)')
plt.subplot(1, 2, 2)
librosa.display.specshow(R_aff, x_axis='time', y_axis='time',
                         cmap='magma_r')
plt.title('Affinity recurrence')
plt.tight_layout()
