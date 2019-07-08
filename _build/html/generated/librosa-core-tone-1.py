# Generate a pure sine tone A4
tone = librosa.tone(440, duration=1)

# Or generate the same signal using `length`
tone = librosa.tone(440, sr=22050, length=22050)

# Display spectrogram

import matplotlib.pyplot as plt
plt.figure()
S = librosa.feature.melspectrogram(y=tone)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         x_axis='time', y_axis='mel')
plt.show()
