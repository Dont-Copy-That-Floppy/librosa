# Apply a standard pre-emphasis filter

import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=10)
y_filt = librosa.effects.preemphasis(y)
# and plot the results for comparison
S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
S_preemph = librosa.amplitude_to_db(np.abs(librosa.stft(y_filt)), ref=np.max)
plt.subplot(2,1,1)
librosa.display.specshow(S_orig, y_axis='log', x_axis='time')
plt.title('Original signal')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(S_preemph, y_axis='log', x_axis='time')
plt.title('Pre-emphasized signal')
plt.colorbar()
plt.tight_layout();

# Apply pre-emphasis in pieces for block streaming.  Note that the second block
# initializes `zi` with the final state `zf` returned by the first call.

y_filt_1, zf = librosa.effects.preemphasis(y[:1000], return_zf=True)
y_filt_2, zf = librosa.effects.preemphasis(y[1000:], zi=zf, return_zf=True)
np.allclose(y_filt, np.concatenate([y_filt_1, y_filt_2]))
# True
