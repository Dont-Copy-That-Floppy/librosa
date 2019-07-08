# Use a shorter window for each filter

basis, lengths = librosa.filters.constant_q(22050, filter_scale=0.5)

# Plot one octave of filters in time and frequency

basis, lengths = librosa.filters.constant_q(22050)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
for i, (f, n) in enumerate(zip(basis, notes)[:12]):
    f_scale = librosa.util.normalize(f) / 2
    plt.plot(i + f_scale.real)
    plt.plot(i + f_scale.imag, linestyle=':')
plt.axis('tight')
plt.yticks(range(len(notes[:12])), notes[:12])
plt.ylabel('CQ filters')
plt.title('CQ filters (one octave, time domain)')
plt.xlabel('Time (samples at 22050 Hz)')
plt.legend(['Real', 'Imaginary'], frameon=True, framealpha=0.8)
plt.subplot(2, 1, 2)
F = np.abs(np.fft.fftn(basis, axes=[-1]))
# Keep only the positive frequencies
F = F[:, :(1 + F.shape[1] // 2)]
librosa.display.specshow(F, x_axis='linear')
plt.yticks(range(len(notes))[::12], notes[::12])
plt.ylabel('CQ filters')
plt.title('CQ filter magnitudes (frequency domain)')
plt.tight_layout()
