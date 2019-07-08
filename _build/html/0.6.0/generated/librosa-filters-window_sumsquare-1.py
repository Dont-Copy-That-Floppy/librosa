# For a fixed frame length (2048), compare modulation effects for a Hann window
# at different hop lengths:

n_frames = 50
wss_256 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=256)
wss_512 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=512)
wss_1024 = librosa.filters.window_sumsquare('hann', n_frames, hop_length=1024)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3,1,1)
plt.plot(wss_256)
plt.title('hop_length=256')
plt.subplot(3,1,2)
plt.plot(wss_512)
plt.title('hop_length=512')
plt.subplot(3,1,3)
plt.plot(wss_1024)
plt.title('hop_length=1024')
plt.tight_layout()
