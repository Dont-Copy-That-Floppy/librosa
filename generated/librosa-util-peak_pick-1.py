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
times = librosa.frames_to_time(np.arange(len(onset_env)),
                               sr=sr, hop_length=512)
plt.figure()
ax = plt.subplot(2, 1, 2)
D = librosa.stft(y)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.subplot(2, 1, 1, sharex=ax)
plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
plt.vlines(times[peaks], 0,
           onset_env.max(), color='r', alpha=0.8,
           label='Selected peaks')
plt.legend(frameon=True, framealpha=0.8)
plt.axis('tight')
plt.tight_layout()
plt.show()
