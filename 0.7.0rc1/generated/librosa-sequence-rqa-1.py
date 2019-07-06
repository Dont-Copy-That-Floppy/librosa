# Simple diagonal path enhancement (L-mode)

import numpy as np
import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.util.example_audio_file(),
                     offset=10, duration=30)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
# Use time-delay embedding to reduce noise
chroma_stack = librosa.feature.stack_memory(chroma, n_steps=3)
# Build recurrence, suppress self-loops within 1 second
rec = librosa.segment.recurrence_matrix(chroma_stack, width=43,
                                        mode='affinity',
                                        metric='cosine')
# using infinite cost for gaps enforces strict path continuation
L_score, L_path = librosa.sequence.rqa(rec, np.inf, np.inf,
                                       knight_moves=False)
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
librosa.display.specshow(rec, x_axis='frames', y_axis='frames')
plt.title('Recurrence matrix')
plt.colorbar()
plt.subplot(1,2,2)
librosa.display.specshow(L_score, x_axis='frames', y_axis='frames')
plt.title('Alignment score matrix')
plt.colorbar()
plt.plot(L_path[:, 1], L_path[:, 0], label='Optimal path', color='c')
plt.legend()
plt.show()

# Full alignment using gaps and knight moves

# New gaps cost 5, extending old gaps cost 10 for each step
score, path = librosa.sequence.rqa(rec, 5, 10)
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
librosa.display.specshow(rec, x_axis='frames', y_axis='frames')
plt.title('Recurrence matrix')
plt.colorbar()
plt.subplot(1,2,2)
librosa.display.specshow(score, x_axis='frames', y_axis='frames')
plt.title('Alignment score matrix')
plt.plot(path[:, 1], path[:, 0], label='Optimal path', color='c')
plt.colorbar()
plt.legend()
plt.show()
