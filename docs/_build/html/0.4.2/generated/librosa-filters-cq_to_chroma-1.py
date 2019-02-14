# Get a CQT, and wrap bins to chroma

y, sr = librosa.load(librosa.util.example_audio_file())
CQT = librosa.cqt(y, sr=sr)
chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0])
chromagram = chroma_map.dot(CQT)
# Max-normalize each time step
chromagram = librosa.util.normalize(chromagram, axis=0)

import matplotlib.pyplot as plt
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.logamplitude(CQT**2,
                                              ref_power=np.max),
                         y_axis='cqt_note', x_axis='time')
plt.title('CQT Power')
plt.colorbar()
plt.subplot(3, 1, 2)
librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
plt.title('Chroma (wrapped CQT)')
plt.colorbar()
plt.subplot(3, 1, 3)
chroma = librosa.feature.chromagram(y=y, sr=sr)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.title('librosa.feature.chroma')
plt.colorbar()
plt.tight_layout()
