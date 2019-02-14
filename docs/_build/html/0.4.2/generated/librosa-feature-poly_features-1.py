y, sr = librosa.load(librosa.util.example_audio_file())
S = np.abs(librosa.stft(y))

# Line features

line = librosa.feature.poly_features(S=S, sr=sr)
line
# array([[ -2.406e-08,  -5.051e-06, ...,  -1.103e-08,  -5.651e-09],
# [  3.445e-04,   3.834e-02, ...,   2.661e-04,   2.239e-04]])

# Quadratic features

quad = librosa.feature.poly_features(S=S, order=2)
quad
# array([[  6.276e-12,   2.010e-09, ...,   1.493e-12,   1.000e-13],
# [ -9.325e-08,  -2.721e-05, ...,  -2.749e-08,  -6.754e-09],
# [  4.715e-04,   7.902e-02, ...,   2.963e-04,   2.259e-04]])

# Plot the results for comparison

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(line)
plt.colorbar()
plt.title('Line coefficients')
plt.subplot(3, 1, 2)
librosa.display.specshow(quad)
plt.colorbar()
plt.title('Quadratic coefficients')
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
                         y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
