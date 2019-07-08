y, sr = librosa.load(librosa.util.example_audio_file())
onset_env = librosa.onset.onset_strength(y, sr=sr)
tempo = librosa.beat.estimate_tempo(onset_env, sr=sr)
tempo
# 103.359375

# Plot the estimated tempo against the onset autocorrelation

import matplotlib.pyplot as plt
# Compute 2-second windowed autocorrelation
hop_length = 512
ac = librosa.autocorrelate(onset_env, 2 * sr // hop_length)
freqs = librosa.tempo_frequencies(len(ac), sr=sr,
                                  hop_length=hop_length)
# Plot on a BPM axis.  We skip the first (0-lag) bin.
plt.figure(figsize=(8,4))
plt.semilogx(freqs[1:], librosa.util.normalize(ac)[1:],
             label='Onset autocorrelation', basex=2)
plt.axvline(tempo, 0, 1, color='r', alpha=0.75, linestyle='--',
           label='Tempo: {:.2f} BPM'.format(tempo))
plt.xlabel('Tempo (BPM)')
plt.grid()
plt.legend(frameon=True)
plt.axis('tight')
