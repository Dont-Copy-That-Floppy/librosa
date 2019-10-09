# Visualize NMF output for a spectrogram S

# Sort the columns of W by peak frequency bin
y, sr = librosa.load(librosa.util.example_audio_file())
S = np.abs(librosa.stft(y))
W, H = librosa.decompose.decompose(S, n_components=32)
W_sort = librosa.util.axis_sort(W)

# Or sort by the lowest frequency bin

W_sort = librosa.util.axis_sort(W, value=np.argmin)

# Or sort the rows instead of the columns

W_sort_rows = librosa.util.axis_sort(W, axis=0)

# Get the sorting index also, and use it to permute the rows of H

W_sort, idx = librosa.util.axis_sort(W, index=True)
H_sort = H[idx, :]

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(W, ref=np.max),
                         y_axis='log')
plt.title('W')
plt.subplot(2, 2, 2)
librosa.display.specshow(H, x_axis='time')
plt.title('H')
plt.subplot(2, 2, 3)
librosa.display.specshow(librosa.amplitude_to_db(W_sort,
                                                 ref=np.max),
                         y_axis='log')
plt.title('W sorted')
plt.subplot(2, 2, 4)
librosa.display.specshow(H_sort, x_axis='time')
plt.title('H sorted')
plt.tight_layout()
plt.show()
