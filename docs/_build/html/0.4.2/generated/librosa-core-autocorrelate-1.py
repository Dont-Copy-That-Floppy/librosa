# Compute full autocorrelation of y

y, sr = librosa.load(librosa.util.example_audio_file())
librosa.autocorrelate(y)
# array([  1.584e+04,   1.580e+04, ...,  -1.154e-10,  -2.725e-13])

# Compute onset strength auto-correlation up to 4 seconds

import matplotlib.pyplot as plt
odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
ac = librosa.autocorrelate(odf, max_size=4* sr / 512)
plt.plot(ac)
plt.title('Auto-correlation')
plt.xlabel('Lag (frames)')
