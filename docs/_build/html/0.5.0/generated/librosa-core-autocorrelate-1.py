# Compute full autocorrelation of y

y, sr = librosa.load(librosa.util.example_audio_file(), offset=20, duration=10)
librosa.autocorrelate(y)
# array([  3.226e+03,   3.217e+03, ...,   8.277e-04,   3.575e-04], dtype=float32)

# Compute onset strength auto-correlation up to 4 seconds

import matplotlib.pyplot as plt
odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
ac = librosa.autocorrelate(odf, max_size=4* sr / 512)
plt.plot(ac)
plt.title('Auto-correlation')
plt.xlabel('Lag (frames)')
