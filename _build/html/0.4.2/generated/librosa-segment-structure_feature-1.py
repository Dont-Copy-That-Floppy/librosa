# Build the structure feature over mfcc similarity

y, sr = librosa.load(librosa.util.example_audio_file())
mfccs = librosa.feature.mfcc(y=y, sr=sr)
recurrence = librosa.segment.recurrence_matrix(mfccs)
struct = librosa.segment.structure_feature(recurrence)

# Invert the structure feature to get a recurrence matrix

recurrence_2 = librosa.segment.structure_feature(struct,
                                                 inverse=True)

# Display recurrence in time-time and time-lag space

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
librosa.display.specshow(recurrence, aspect='equal', x_axis='time',
                         y_axis='time')
plt.ylabel('Time')
plt.title('Recurrence (time-time)')
plt.subplot(1, 2, 2)
librosa.display.specshow(struct, aspect='auto', x_axis='time')
plt.ylabel('Lag')
plt.title('Structure feature')
plt.tight_layout()
