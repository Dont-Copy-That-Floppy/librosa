# Visualize an STFT power spectrum

import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.util.example_audio_file())
plt.figure(figsize=(12, 8))

D = librosa.logamplitude(np.abs(librosa.stft(y))**2, ref_power=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

# Or on a logarithmic scale

plt.subplot(4, 2, 2)
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')

# Or use a CQT scale

CQT = librosa.logamplitude(librosa.cqt(y, sr=sr)**2, ref_power=np.max)
plt.subplot(4, 2, 3)
librosa.display.specshow(CQT, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')

plt.subplot(4, 2, 4)
librosa.display.specshow(CQT, y_axis='cqt_hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (Hz)')

# Draw a chromagram with pitch classes

C = librosa.feature.chroma_cqt(y=y, sr=sr)
plt.subplot(4, 2, 5)
librosa.display.specshow(C, y_axis='chroma')
plt.colorbar()
plt.title('Chromagram')

# Force a grayscale colormap (white -> black)

plt.subplot(4, 2, 6)
librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear power spectrogram (grayscale)')

# Draw time markers automatically

plt.subplot(4, 2, 7)
librosa.display.specshow(D, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log power spectrogram')

# Draw a tempogram with BPM markers

plt.subplot(4, 2, 8)
oenv = librosa.onset.onset_strength(y=y, sr=sr)
tempo = librosa.beat.estimate_tempo(oenv, sr=sr)
Tgram = librosa.feature.tempogram(y=y, sr=sr)
librosa.display.specshow(Tgram[:100], x_axis='time', y_axis='tempo',
                         tmin=tempo/4, tmax=tempo*2, n_yticks=4)
plt.colorbar()
plt.title('Tempogram')
plt.tight_layout()
