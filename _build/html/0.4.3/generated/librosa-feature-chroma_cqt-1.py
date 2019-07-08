# Compare a long-window STFT chromagram to the CQT chromagram

y, sr = librosa.load(librosa.util.example_audio_file(),
                     offset=10, duration=15)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_chroma=12, n_fft=4096)
chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(chroma_stft, y_axis='chroma')
plt.title('chroma_stft')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
plt.title('chroma_cqt')
plt.colorbar()
plt.tight_layout()
