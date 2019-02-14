# Beat-synchronous CQT spectra

y, sr = librosa.load(librosa.util.example_audio_file())
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
cqt = librosa.cqt(y=y, sr=sr)

# By default, use mean aggregation

cqt_avg = librosa.util.sync(cqt, beats)

# Use median-aggregation instead of mean

cqt_med = librosa.util.sync(cqt, beats,
                               aggregate=np.median)

# Or sub-beat synchronization

sub_beats = librosa.segment.subsegment(cqt, beats)
cqt_med_sub = librosa.util.sync(cqt, sub_beats, aggregate=np.median)

# Plot the results

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.logamplitude(cqt**2,
                                              ref_power=np.max),
                         x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('CQT power, shape={}'.format(cqt.shape))
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.logamplitude(cqt_med**2,
                                              ref_power=np.max))
plt.colorbar(format='%+2.0f dB')
plt.title('Beat synchronous CQT power, '
          'shape={}'.format(cqt_med.shape))
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.logamplitude(cqt_med_sub**2,
                                              ref_power=np.max))
plt.colorbar(format='%+2.0f dB')
plt.title('Sub-beat synchronous CQT power, '
          'shape={}'.format(cqt_med_sub.shape))
plt.tight_layout()
