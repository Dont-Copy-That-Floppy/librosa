# Generate a signal and time-stretch it (with energy normalization)
scale = 1.25
freq = 3.0
x1 = np.linspace(0, 1, num=1024, endpoint=False)
x2 = np.linspace(0, 1, num=scale * len(x1), endpoint=False)
y1 = np.sin(2 * np.pi * freq * x1)
y2 = np.sin(2 * np.pi * freq * x2) / np.sqrt(scale)
# Verify that the two signals have the same energy
np.sum(np.abs(y1)**2), np.sum(np.abs(y2)**2)
# (255.99999999999997, 255.99999999999969)
scale1 = librosa.fmt(y1, n_fmt=512)
scale2 = librosa.fmt(y2, n_fmt=512)
# And plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(y1, label='Original')
plt.plot(y2, linestyle='--', label='Stretched')
plt.xlabel('time (samples)')
plt.title('Input signals')
plt.legend(frameon=True)
plt.axis('tight')
plt.subplot(1, 2, 2)
plt.semilogy(np.abs(scale1), label='Original')
plt.semilogy(np.abs(scale2), linestyle='--', label='Stretched')
plt.xlabel('scale coefficients')
plt.title('Scale transform magnitude')
plt.legend(frameon=True)
plt.axis('tight')
plt.tight_layout()

# Plot the scale transform of an onset strength autocorrelation
y, sr = librosa.load(librosa.util.example_audio_file(),
                     offset=10.0, duration=30.0)
odf = librosa.onset.onset_strength(y=y, sr=sr)
# Auto-correlate with up to 10 seconds lag
odf_ac = librosa.autocorrelate(odf, max_size=10 * sr // 512)
# Normalize
odf_ac = librosa.util.normalize(odf_ac, norm=np.inf)
# Compute the scale transform
odf_ac_scale = librosa.fmt(librosa.util.normalize(odf_ac), n_fmt=512)
# Plot the results
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(odf, label='Onset strength')
plt.axis('tight')
plt.xlabel('Time (frames)')
plt.xticks([])
plt.legend(frameon=True)
plt.subplot(3, 1, 2)
plt.plot(odf_ac, label='Onset autocorrelation')
plt.axis('tight')
plt.xlabel('Lag (frames)')
plt.xticks([])
plt.legend(frameon=True)
plt.subplot(3, 1, 3)
plt.semilogy(np.abs(odf_ac_scale), label='Scale transform magnitude')
plt.axis('tight')
plt.xlabel('scale coefficients')
plt.legend(frameon=True)
plt.tight_layout()
