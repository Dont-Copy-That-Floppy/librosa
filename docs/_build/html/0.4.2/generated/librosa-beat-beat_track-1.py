# Track beats using time series input

y, sr = librosa.load(librosa.util.example_audio_file())

tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
tempo
# 129.19921875

# Print the first 20 beat frames

beats[:20]
# array([ 461,  500,  540,  580,  619,  658,  698,  737,  777,
# 817,  857,  896,  936,  976, 1016, 1055, 1095, 1135,
# 1175, 1214])

# Or print them as timestamps

librosa.frames_to_time(beats[:20], sr=sr)
# array([ 0.093,  0.534,  0.998,  1.463,  1.927,  2.368,  2.833,
# 3.297,  3.762,  4.203,  4.667,  5.132,  5.596,  6.06 ,
# 6.525,  6.989,  7.454,  7.918,  8.382,  8.847])

# Track beats using a pre-computed onset envelope

onset_env = librosa.onset.onset_strength(y, sr=sr,
                                         aggregate=np.median)
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,
                                       sr=sr)
tempo
# 64.599609375
beats[:20]
# array([ 461,  500,  540,  580,  619,  658,  698,  737,  777,
# 817,  857,  896,  936,  976, 1016, 1055, 1095, 1135,
# 1175, 1214])

# Plot the beat events against the onset strength envelope

import matplotlib.pyplot as plt
hop_length = 512
plt.figure()
plt.plot(librosa.util.normalize(onset_env), label='Onset strength')
plt.vlines(beats, 0, 1, alpha=0.5, color='r',
           linestyle='--', label='Beats')
plt.legend(frameon=True, framealpha=0.75)
# Limit the plot to a 15-second window
plt.xlim([10 * sr / hop_length, 25 * sr / hop_length])
plt.xticks(np.linspace(10, 25, 5) * sr / hop_length,
           np.linspace(10, 25, 5))
plt.xlabel('Time (s)')
plt.tight_layout()
