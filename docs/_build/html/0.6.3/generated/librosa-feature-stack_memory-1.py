# Keep two steps (current and previous)

data = np.arange(-3, 3)
librosa.feature.stack_memory(data)
# array([[-3, -2, -1,  0,  1,  2],
# [ 0, -3, -2, -1,  0,  1]])

# Or three steps

librosa.feature.stack_memory(data, n_steps=3)
# array([[-3, -2, -1,  0,  1,  2],
# [ 0, -3, -2, -1,  0,  1],
# [ 0,  0, -3, -2, -1,  0]])

# Use reflection padding instead of zero-padding

librosa.feature.stack_memory(data, n_steps=3, mode='reflect')
# array([[-3, -2, -1,  0,  1,  2],
# [-2, -3, -2, -1,  0,  1],
# [-1, -2, -3, -2, -1,  0]])

# Or pad with edge-values, and delay by 2

librosa.feature.stack_memory(data, n_steps=3, delay=2, mode='edge')
# array([[-3, -2, -1,  0,  1,  2],
# [-3, -3, -3, -2, -1,  0],
# [-3, -3, -3, -3, -3, -2]])

# Stack time-lagged beat-synchronous chroma edge padding

y, sr = librosa.load(librosa.util.example_audio_file())
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
beats = librosa.util.fix_frames(beats, x_min=0, x_max=chroma.shape[1])
chroma_sync = librosa.util.sync(chroma, beats)
chroma_lag = librosa.feature.stack_memory(chroma_sync, n_steps=3,
                                          mode='edge')

# Plot the result

import matplotlib.pyplot as plt
beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
librosa.display.specshow(chroma_lag, y_axis='chroma', x_axis='time',
                         x_coords=beat_times)
plt.yticks([0, 12, 24], ['Lag=0', 'Lag=1', 'Lag=2'])
plt.title('Time-lagged chroma')
plt.colorbar()
plt.tight_layout()
