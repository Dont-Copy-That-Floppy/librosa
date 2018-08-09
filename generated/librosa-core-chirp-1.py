# Generate a exponential chirp from A4 to A5
exponential_chirp = librosa.chirp(440, 880, duration=1)

# Or generate the same signal using `length`
exponential_chirp = librosa.chirp(440, 880, sr=22050, length=22050)

# Or generate a linear chirp instead
linear_chirp = librosa.chirp(440, 880, duration=1, linear=True)

# Display spectrogram for both exponential and linear chirps

import matplotlib.pyplot as plt
plt.figure()
S_exponential = librosa.feature.melspectrogram(y=exponential_chirp)
ax = plt.subplot(2,1,1)
librosa.display.specshow(librosa.power_to_db(S_exponential, ref=np.max),
                         x_axis='time', y_axis='mel')
plt.subplot(2,1,2, sharex=ax)
S_linear = librosa.feature.melspectrogram(y=linear_chirp)
librosa.display.specshow(librosa.power_to_db(S_linear, ref=np.max),
                         x_axis='time', y_axis='mel')
plt.tight_layout()
