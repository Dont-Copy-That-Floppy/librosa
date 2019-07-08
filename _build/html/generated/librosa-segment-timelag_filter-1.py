# Apply a 5-bin median filter to the diagonal of a recurrence matrix

y, sr = librosa.load(librosa.util.example_audio_file())
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
rec = librosa.segment.recurrence_matrix(chroma)
from scipy.ndimage import median_filter
diagonal_median = librosa.segment.timelag_filter(median_filter)
rec_filtered = diagonal_median(rec, size=(1, 3), mode='mirror')

# Or with affinity weights

rec_aff = librosa.segment.recurrence_matrix(chroma, mode='affinity')
rec_aff_fil = diagonal_median(rec_aff, size=(1, 3), mode='mirror')

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.subplot(2, 2, 1)
librosa.display.specshow(rec, y_axis='time')
plt.title('Raw recurrence matrix')
plt.subplot(2, 2, 2)
librosa.display.specshow(rec_filtered)
plt.title('Filtered recurrence matrix')
plt.subplot(2, 2, 3)
librosa.display.specshow(rec_aff, x_axis='time', y_axis='time',
                         cmap='magma_r')
plt.title('Raw affinity matrix')
plt.subplot(2, 2, 4)
librosa.display.specshow(rec_aff_fil, x_axis='time',
                         cmap='magma_r')
plt.title('Filtered affinity matrix')
plt.tight_layout()
plt.show()
