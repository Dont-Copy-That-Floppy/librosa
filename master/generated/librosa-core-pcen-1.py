# Compare PCEN to log amplitude (dB) scaling on Mel spectra

import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.util.example_audio_file(),
                     offset=10, duration=10)

# We'll use power=1 to get a magnitude spectrum
# instead of a power spectrum
S = librosa.feature.melspectrogram(y, sr=sr, power=1)
log_S = librosa.amplitude_to_db(S, ref=np.max)
pcen_S = librosa.pcen(S)
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(log_S, x_axis='time', y_axis='mel')
plt.title('log amplitude (dB)')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel')
plt.title('Per-channel energy normalization')
plt.colorbar()
plt.tight_layout()

# Compare PCEN with and without max-filtering

pcen_max = librosa.pcen(S, max_size=3)
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel')
plt.title('Per-channel energy normalization (no max-filter)')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(pcen_max, x_axis='time', y_axis='mel')
plt.title('Per-channel energy normalization (max_size=3)')
plt.colorbar()
plt.tight_layout()
