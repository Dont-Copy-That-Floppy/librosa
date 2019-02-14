# Compute MFCC deltas, delta-deltas

y, sr = librosa.load(librosa.util.example_audio_file())
mfcc = librosa.feature.mfcc(y=y, sr=sr)
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta
# array([[  2.929e+01,   3.090e+01, ...,   0.000e+00,   0.000e+00],
# [  2.226e+01,   2.553e+01, ...,   3.944e-31,   3.944e-31],
# ...,
# [ -1.192e+00,  -6.099e-01, ...,   9.861e-32,   9.861e-32],
# [ -5.349e-01,  -2.077e-01, ...,   1.183e-30,   1.183e-30]])
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
mfcc_delta2
# array([[  1.281e+01,   1.020e+01, ...,   0.000e+00,   0.000e+00],
# [  2.726e+00,   3.558e+00, ...,   0.000e+00,   0.000e+00],
# ...,
# [ -1.702e-01,  -1.509e-01, ...,   0.000e+00,   0.000e+00],
# [ -9.021e-02,  -7.007e-02, ...,  -2.190e-47,  -2.190e-47]])

import matplotlib.pyplot as plt
plt.subplot(3, 1, 1)
librosa.display.specshow(mfcc)
plt.title('MFCC')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(mfcc_delta)
plt.title(r'MFCC-$\Delta$')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(mfcc_delta2, x_axis='time')
plt.title(r'MFCC-$\Delta^2$')
plt.colorbar()
plt.tight_layout()
