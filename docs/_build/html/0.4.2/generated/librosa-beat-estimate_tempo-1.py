y, sr = librosa.load(librosa.util.example_audio_file())
onset_env = librosa.onset.onset_strength(y, sr=sr)
tempo = librosa.beat.estimate_tempo(onset_env, sr=sr)
tempo
# 129.19921875

# Plot the estimated tempo against the onset autocorrelation

import matplotlib.pyplot as plt
# Compute 2-second windowed autocorrelation
hop_length = 512
ac = librosa.autocorrelate(onset_env, 2 * sr // hop_length)
# Convert tempo estimate from bpm to frames
tempo_frames = (60 * sr / hop_length) / tempo
plt.plot(librosa.util.normalize(ac),
         label='Onset autocorrelation')
plt.vlines([tempo_frames], 0, 1,
           color='r', alpha=0.75, linestyle='--',
           label='Tempo: {:.2f} BPM'.format(tempo))
librosa.display.time_ticks(librosa.frames_to_time(np.arange(len(ac)),
                                                  sr=sr))
plt.xlabel('Lag')
plt.legend()
plt.axis('tight')
