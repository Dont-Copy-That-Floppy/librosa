# Build a simple chroma filter bank

chromafb = librosa.filters.chroma(22050, 4096)
# array([[  1.689e-05,   3.024e-04, ...,   4.639e-17,   5.327e-17],
# [  1.716e-05,   2.652e-04, ...,   2.674e-25,   3.176e-25],
# ...,
# [  1.578e-05,   3.619e-04, ...,   8.577e-06,   9.205e-06],
# [  1.643e-05,   3.355e-04, ...,   1.474e-10,   1.636e-10]])

# Use quarter-tones instead of semitones

librosa.filters.chroma(22050, 4096, n_chroma=24)
# array([[  1.194e-05,   2.138e-04, ...,   6.297e-64,   1.115e-63],
# [  1.206e-05,   2.009e-04, ...,   1.546e-79,   2.929e-79],
# ...,
# [  1.162e-05,   2.372e-04, ...,   6.417e-38,   9.923e-38],
# [  1.180e-05,   2.260e-04, ...,   4.697e-50,   7.772e-50]])

# Equally weight all octaves

librosa.filters.chroma(22050, 4096, octwidth=None)
# array([[  3.036e-01,   2.604e-01, ...,   2.445e-16,   2.809e-16],
# [  3.084e-01,   2.283e-01, ...,   1.409e-24,   1.675e-24],
# ...,
# [  2.836e-01,   3.116e-01, ...,   4.520e-05,   4.854e-05],
# [  2.953e-01,   2.888e-01, ...,   7.768e-10,   8.629e-10]])

import matplotlib.pyplot as plt
plt.figure()
librosa.display.specshow(chromafb, x_axis='linear')
plt.ylabel('Chroma filter')
plt.title('Chroma filter bank')
plt.colorbar()
plt.tight_layout()
