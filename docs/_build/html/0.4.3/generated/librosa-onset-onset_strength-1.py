# First, load some audio and plot the spectrogram

import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.util.example_audio_file(),
                     duration=10.0)
D = np.abs(librosa.stft(y))**2
plt.figure()
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max),
                         y_axis='log')
plt.title('Power spectrogram')

# Construct a standard onset function

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
plt.subplot(2, 1, 2)
plt.plot(2 + onset_env / onset_env.max(), alpha=0.8,
         label='Mean aggregation (mel)')

# Median aggregation, and custom mel options

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         aggregate=np.median,
                                         fmax=8000, n_mels=256)
plt.plot(1 + onset_env / onset_env.max(), alpha=0.8,
         label='Median aggregation (custom mel)')

# Constant-Q spectrogram instead of Mel

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         feature=librosa.cqt)
plt.plot(onset_env / onset_env.max(), alpha=0.8,
         label='Mean aggregation (CQT)')

plt.legend(frameon=True, framealpha=0.75)
librosa.display.time_ticks(librosa.frames_to_time(np.arange(len(onset_env))))
plt.ylabel('Normalized strength')
plt.yticks([])
plt.axis('tight')
plt.tight_layout()
