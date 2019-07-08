y, sr = librosa.load(librosa.util.example_audio_file())
hop_length = 1024
mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
recurrence = librosa.segment.recurrence_matrix(mfccs)
lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=True)
lag_nopad = librosa.segment.recurrence_to_lag(recurrence, pad=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(lag_pad, x_axis='time', y_axis='lag',
                         hop_length=hop_length)
plt.title('Lag (zero-padded)')
plt.subplot(1, 2, 2)
librosa.display.specshow(lag_nopad, x_axis='time', hop_length=hop_length)
plt.title('Lag (no padding)')
plt.tight_layout()
plt.show()
