# Beat-synchronous CQT spectra

y, sr = librosa.load(librosa.util.example_audio_file())
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
C = np.abs(librosa.cqt(y=y, sr=sr))
beats = librosa.util.fix_frames(beats, x_max=C.shape[1])

# By default, use mean aggregation

C_avg = librosa.util.sync(C, beats)

# Use median-aggregation instead of mean

C_med = librosa.util.sync(C, beats,
                            aggregate=np.median)

# Or sub-beat synchronization

sub_beats = librosa.segment.subsegment(C, beats)
sub_beats = librosa.util.fix_frames(sub_beats, x_max=C.shape[1])
C_med_sub = librosa.util.sync(C, sub_beats, aggregate=np.median)

# Plot the results

import matplotlib.pyplot as plt
beat_t = librosa.frames_to_time(beats, sr=sr)
subbeat_t = librosa.frames_to_time(sub_beats, sr=sr)
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(C,
                                                 ref=np.max),
                         x_axis='time')
plt.title('CQT power, shape={}'.format(C.shape))
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(C_med,
                                                 ref=np.max),
                         x_coords=beat_t, x_axis='time')
plt.title('Beat synchronous CQT power, '
          'shape={}'.format(C_med.shape))
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(C_med_sub,
                                                 ref=np.max),
                         x_coords=subbeat_t, x_axis='time')
plt.title('Sub-beat synchronous CQT power, '
          'shape={}'.format(C_med_sub.shape))
plt.tight_layout()
plt.show()
