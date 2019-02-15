import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
semitone_filterbank, sample_rates = librosa.filters.semitone_filterbank()
plt.figure(figsize=(10, 6))
for cur_sr, cur_filter in zip(sample_rates, semitone_filterbank):
   w, h = scipy.signal.freqz(cur_filter[0], cur_filter[1], worN=2000)
   plt.plot((cur_sr / (2 * np.pi)) * w, 20 * np.log10(abs(h)))
plt.semilogx()
plt.xlim([20, 10e3])
plt.ylim([-60, 3])
plt.title('Magnitude Responses of the Pitch Filterbank')
plt.xlabel('Log-Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.tight_layout()
