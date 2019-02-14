# Simple log frequency filters

logfb = librosa.filters.logfrequency(22050, 4096)
logfb
# array([[ 0.,  0., ...,  0.,  0.],
# [ 0.,  0., ...,  0.,  0.],
# ...,
# [ 0.,  0., ...,  0.,  0.],
# [ 0.,  0., ...,  0.,  0.]])

# Use a narrower frequency range

librosa.filters.logfrequency(22050, 4096, n_bins=48, fmin=110)
# array([[ 0.,  0., ...,  0.,  0.],
# [ 0.,  0., ...,  0.,  0.],
# ...,
# [ 0.,  0., ...,  0.,  0.],
# [ 0.,  0., ...,  0.,  0.]])

# Use narrower filters for sparser response: 5% of a semitone

librosa.filters.logfrequency(22050, 4096, spread=0.05)

# Or wider: 50% of a semitone

librosa.filters.logfrequency(22050, 4096, spread=0.5)

import matplotlib.pyplot as plt
plt.figure()
librosa.display.specshow(logfb, x_axis='linear')
plt.ylabel('Logarithmic filters')
plt.title('Log-frequency filter bank')
plt.colorbar()
plt.tight_layout()
