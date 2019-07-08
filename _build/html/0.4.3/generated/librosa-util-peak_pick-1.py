y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         hop_length=512,
                                         aggregate=np.median)
peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
peaks
# array([  4,  23,  73, 102, 142, 162, 182, 211, 261, 301, 320,
# 331, 348, 368, 382, 396, 411, 431, 446, 461, 476, 491,
# 510, 525, 536, 555, 570, 590, 609, 625, 639])

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(onset_env[:30 * sr // 512], alpha=0.8, label='Onset strength')
plt.vlines(peaks[peaks < 30 * sr // 512], 0,
           onset_env.max(), color='r', alpha=0.8,
           label='Selected peaks')
plt.legend(frameon=True, framealpha=0.8)
plt.axis('tight')
plt.axis('off')
plt.subplot(2, 1, 2)
D = np.abs(librosa.stft(y))**2
librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max),
                         y_axis='log', x_axis='time')
plt.tight_layout()
