# Visualize the PLP compared to an onset strength envelope.
# Both are normalized here to make comparison easier.

y, sr = librosa.load(librosa.util.example_audio_file())
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
melspec = librosa.feature.melspectrogram(y=y, sr=sr)
import matplotlib.pyplot as plt
ax = plt.subplot(2,1,1)
librosa.display.specshow(librosa.power_to_db(melspec,
                                             ref=np.max),
                         x_axis='time', y_axis='mel')
plt.title('Mel spectrogram')
plt.subplot(2,1,2, sharex=ax)
plt.plot(librosa.times_like(onset_env),
         librosa.util.normalize(onset_env),
         label='Onset strength')
plt.plot(librosa.times_like(pulse),
         librosa.util.normalize(pulse),
         label='Predominant local pulse (PLP)')
plt.legend()
plt.xlim([30, 35])
plt.tight_layout()
plt.show()

# PLP local maxima can be used as estimates of beat positions.

tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
import matplotlib.pyplot as plt
ax = plt.subplot(2,1,1)
times = librosa.times_like(onset_env, sr=sr)
plt.plot(times, librosa.util.normalize(onset_env),
         label='Onset strength')
plt.vlines(times[beats], 0, 1, alpha=0.5, color='r',
           linestyle='--', label='Beats')
plt.legend(frameon=True, framealpha=0.75)
plt.title('librosa.beat.beat_track')
# Limit the plot to a 15-second window
plt.subplot(2,1,2, sharex=ax)
times = librosa.times_like(pulse, sr=sr)
plt.plot(times, librosa.util.normalize(pulse),
         label='PLP')
plt.vlines(times[beats_plp], 0, 1, alpha=0.5, color='r',
           linestyle='--', label='PLP Beats')
plt.legend(frameon=True, framealpha=0.75)
plt.title('librosa.beat.plp')
plt.xlim(30, 35)
ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
plt.tight_layout()
plt.show()
