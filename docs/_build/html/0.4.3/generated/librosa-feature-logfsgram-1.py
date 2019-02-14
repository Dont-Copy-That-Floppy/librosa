# From time-series input

y, sr = librosa.load(librosa.util.example_audio_file())
L = librosa.feature.logfsgram(y=y, sr=sr)
L
# array([[  1.309e-02,   1.228e+00, ...,   3.785e-08,   7.624e-09],
# [  1.630e-24,   1.528e-22, ...,   4.710e-30,   9.488e-31],
# ...,
# [  2.617e-05,   3.807e-04, ...,   6.387e-08,   6.000e-08],
# [  3.214e-05,   3.814e-04, ...,   7.599e-08,   6.046e-08]])

# Plot the pseudo CQT

import matplotlib.pyplot as plt
plt.figure()
librosa.display.specshow(librosa.logamplitude(L,
                                              ref_power=np.max),
                         y_axis='cqt_hz', x_axis='time')
plt.title('Log-frequency power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
