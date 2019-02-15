n_fft = 2048
dct_filters = librosa.filters.dct(13, 1 + n_fft // 2)
dct_filters
# array([[ 0.031,  0.031, ...,  0.031,  0.031],
# [ 0.044,  0.044, ..., -0.044, -0.044],
# ...,
# [ 0.044,  0.044, ..., -0.044, -0.044],
# [ 0.044,  0.044, ...,  0.044,  0.044]])

import matplotlib.pyplot as plt
plt.figure()
librosa.display.specshow(dct_filters, x_axis='linear')
plt.ylabel('DCT function')
plt.title('DCT filter bank')
plt.colorbar()
plt.tight_layout()
