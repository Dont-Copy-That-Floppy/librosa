# Separate into harmonic and percussive

y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
D = librosa.stft(y)
H, P = librosa.decompose.hpss(D)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),
                                                 ref=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Full power spectrogram')
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(H),
                                                 ref=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Harmonic power spectrogram')
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(P),
                                                 ref=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Percussive power spectrogram')
plt.tight_layout()
plt.show()

# Or with a narrower horizontal filter

H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

# Just get harmonic/percussive masks, not the spectra

mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
mask_H
# array([[  1.000e+00,   1.469e-01, ...,   2.648e-03,   2.164e-03],
# [  1.000e+00,   2.368e-01, ...,   9.413e-03,   7.703e-03],
# ...,
# [  8.869e-01,   5.673e-02, ...,   4.603e-02,   1.247e-05],
# [  7.068e-01,   2.194e-02, ...,   4.453e-02,   1.205e-05]], dtype=float32)
mask_P
# array([[  2.858e-05,   8.531e-01, ...,   9.974e-01,   9.978e-01],
# [  1.586e-05,   7.632e-01, ...,   9.906e-01,   9.923e-01],
# ...,
# [  1.131e-01,   9.433e-01, ...,   9.540e-01,   1.000e+00],
# [  2.932e-01,   9.781e-01, ...,   9.555e-01,   1.000e+00]], dtype=float32)

# Separate into harmonic/percussive/residual components by using a margin > 1.0

H, P = librosa.decompose.hpss(D, margin=3.0)
R = D - (H+P)
y_harm = librosa.core.istft(H)
y_perc = librosa.core.istft(P)
y_resi = librosa.core.istft(R)

# Get a more isolated percussive component by widening its margin

H, P = librosa.decompose.hpss(D, margin=(1.0,5.0))
