# Apply a 5-bin median filter to the diagonal of a recurrence matrix

y, sr = librosa.load(librosa.util.example_audio_file())
mfcc = librosa.feature.mfcc(y=y, sr=sr)
rec = librosa.segment.recurrence_matrix(mfcc, sym=True)
from scipy.ndimage import median_filter
diagonal_median = librosa.segment.timelag_filter(median_filter)
rec_filtered = diagonal_median(rec, size=(1, 5), mode='mirror')

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
librosa.display.specshow(rec, x_axis='time', y_axis='time',
                         aspect='equal')
plt.title('Raw recurrence matrix')
plt.subplot(1, 2, 2)
librosa.display.specshow(rec_filtered, x_axis='time', y_axis='time',
                         aspect='equal')
plt.title('Filtered recurrence matrix')
plt.tight_layout()
