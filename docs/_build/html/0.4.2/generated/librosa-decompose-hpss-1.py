# Separate into harmonic and percussive

y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
D = librosa.stft(y)
H, P = librosa.decompose.hpss(D)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.logamplitude(np.abs(D)**2,
                                              ref_power=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Full power spectrogram')
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.logamplitude(np.abs(H)**2,
                         ref_power=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Harmonic power spectrogram')
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.logamplitude(np.abs(P)**2,
                         ref_power=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Percussive power spectrogram')
plt.tight_layout()

# Or with a narrower horizontal filter

H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

# Just get harmonic/percussive masks, not the spectra

mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
mask_H
# array([[ 1.,  0., ...,  0.,  0.],
# [ 1.,  0., ...,  0.,  0.],
# ...,
# [ 0.,  0., ...,  0.,  0.],
# [ 0.,  0., ...,  0.,  0.]])
mask_P
# array([[ 0.,  1., ...,  1.,  1.],
# [ 0.,  1., ...,  1.,  1.],
# ...,
# [ 1.,  1., ...,  1.,  1.],
# [ 1.,  1., ...,  1.,  1.]])
