# Compute MFCC deltas, delta-deltas

y, sr = librosa.load(librosa.util.example_audio_file())
mfcc = librosa.feature.mfcc(y=y, sr=sr)
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta
# array([[  1.666e+01,   1.666e+01, ...,   1.869e-15,   1.869e-15],
# [  1.784e+01,   1.784e+01, ...,   6.085e-31,   6.085e-31],
# ...,
# [  7.262e-01,   7.262e-01, ...,   9.259e-31,   9.259e-31],
# [  6.578e-01,   6.578e-01, ...,   7.597e-31,   7.597e-31]])

mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
mfcc_delta2
# array([[ -1.703e+01,  -1.703e+01, ...,   3.834e-14,   3.834e-14],
# [ -1.108e+01,  -1.108e+01, ...,  -1.068e-30,  -1.068e-30],
# ...,
# [  4.075e-01,   4.075e-01, ...,  -1.565e-30,  -1.565e-30],
# [  1.676e-01,   1.676e-01, ...,  -2.104e-30,  -2.104e-30]])

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
plt.show()
