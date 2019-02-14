# Re-weight a CQT power spectrum, using peak power as reference

y, sr = librosa.load(librosa.util.example_audio_file())
CQT = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1'))
freqs = librosa.cqt_frequencies(CQT.shape[0],
                                fmin=librosa.note_to_hz('A1'))
perceptual_CQT = librosa.perceptual_weighting(CQT**2,
                                              freqs,
                                              ref=np.max)
perceptual_CQT
# array([[ -80.076,  -80.049, ..., -104.735, -104.735],
# [ -78.344,  -78.555, ..., -103.725, -103.725],
# ...,
# [ -76.272,  -76.272, ...,  -76.272,  -76.272],
# [ -76.485,  -76.485, ...,  -76.485,  -76.485]])

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(CQT,
                                                 ref=np.max),
                         fmin=librosa.note_to_hz('A1'),
                         y_axis='cqt_hz')
plt.title('Log CQT power')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 1, 2)
librosa.display.specshow(perceptual_CQT, y_axis='cqt_hz',
                         fmin=librosa.note_to_hz('A1'),
                         x_axis='time')
plt.title('Perceptually weighted log CQT')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
