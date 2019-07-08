# Cluster by chroma similarity, break into 20 segments

y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
boundary_frames = librosa.segment.agglomerative(chroma, 20)
librosa.frames_to_time(boundary_frames, sr=sr)
# array([  0.   ,   1.672,   2.322,   2.624,   3.251,   3.506,
# 4.18 ,   5.387,   6.014,   6.293,   6.943,   7.198,
# 7.848,   9.033,   9.706,   9.961,  10.635,  10.89 ,
# 11.54 ,  12.539])

# Plot the segmentation against the spectrogram

import matplotlib.pyplot as plt
plt.figure()
S = np.abs(librosa.stft(y))**2
librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max),
                         y_axis='log', x_axis='time')
plt.vlines(boundary_frames, 0, S.shape[0], color='r', alpha=0.9,
           label='Segment boundaries')
plt.legend(frameon=True, shadow=True)
plt.title('Power spectrogram')
plt.tight_layout()
