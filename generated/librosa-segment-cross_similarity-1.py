# Find nearest neighbors in MFCC space between two sequences

y_ref, sr = librosa.load(librosa.util.example_audio_file())
y_comp, sr = librosa.load(librosa.util.example_audio_file(), offset=10)
mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr)
mfcc_comp = librosa.feature.mfcc(y=y_comp, sr=sr)
xsim = librosa.segment.cross_similarity(mfcc_comp, mfcc_ref)

# Or fix the number of nearest neighbors to 5

xsim = librosa.segment.cross_similarity(mfcc_comp, mfcc_ref, k=5)

# Use cosine similarity instead of Euclidean distance

xsim = librosa.segment.cross_similarity(mfcc_comp, mfcc_ref, metric='cosine')

# Use an affinity matrix instead of binary connectivity

xsim_aff = librosa.segment.cross_similarity(mfcc_comp, mfcc_ref, mode='affinity')

# Plot the feature and recurrence matrices

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(xsim, x_axis='time', y_axis='time')
plt.title('Binary recurrence (symmetric)')
plt.subplot(1, 2, 2)
librosa.display.specshow(xsim_aff, x_axis='time', y_axis='time',
                         cmap='magma_r')
plt.title('Affinity recurrence')
plt.tight_layout()
