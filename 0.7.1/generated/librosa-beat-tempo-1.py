# Estimate a static tempo
y, sr = librosa.load(librosa.util.example_audio_file())
onset_env = librosa.onset.onset_strength(y, sr=sr)
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
tempo
# array([129.199])

# Or a static tempo with a uniform prior instead
import scipy.stats
prior = scipy.stats.uniform(30, 300)  # uniform over 30-300 BPM
utempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, prior=prior)
utempo
# array([64.6])

# Or a dynamic tempo
dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                            aggregate=None)
dtempo
# array([ 143.555,  143.555,  143.555, ...,  161.499,  161.499,
# 172.266])

# Dynamic tempo with a proper log-normal prior
prior_lognorm = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
dtempo_lognorm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                                    aggregate=None,
                                    prior=prior_lognorm)
dtempo_lognorm
# array([ 86.133,  86.133, ..., 129.199, 129.199])

# Plot the estimated tempo against the onset autocorrelation

import matplotlib.pyplot as plt
# Convert to scalar
tempo = tempo.item()
utempo = utempo.item()
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
            label='Tempo (default prior): {:.2f} BPM'.format(tempo))
plt.axvline(utempo, 0, 1, color='y', alpha=0.75, linestyle=':',
            label='Tempo (uniform prior): {:.2f} BPM'.format(utempo))
plt.xlabel('Tempo (BPM)')
plt.grid()
plt.title('Static tempo estimation')
plt.legend(frameon=True)
plt.axis('tight')
plt.show()

# Plot dynamic tempo estimates over a tempogram

plt.figure()
tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
                               hop_length=hop_length)
librosa.display.specshow(tg, x_axis='time', y_axis='tempo')
plt.plot(librosa.times_like(dtempo), dtempo,
         color='w', linewidth=1.5, label='Tempo estimate (default prior)')
plt.plot(librosa.times_like(dtempo_lognorm), dtempo_lognorm,
         color='c', linewidth=1.5, linestyle='--',
         label='Tempo estimate (lognorm prior)')
plt.title('Dynamic tempo estimation')
plt.legend(frameon=True, framealpha=0.75)
